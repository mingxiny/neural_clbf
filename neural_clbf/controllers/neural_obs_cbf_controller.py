import itertools
from typing import Tuple, List, Optional
from collections import OrderedDict
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import pytorch_lightning as pl

from neural_clbf.systems import ControlAffineSystem
from neural_clbf.systems.utils import ScenarioList
from neural_clbf.controllers.cbf_controller import CBFController
from neural_clbf.controllers.controller_utils import normalize_with_angles
from neural_clbf.datamodules.episodic_datamodule import EpisodicDataModule
from neural_clbf.experiments import ExperimentSuite


class NeuralObsCBFController(pl.LightningModule, CBFController):
    """
    A neural CBF controller. Differs from the CBFController in that it uses a
    neural network to learn the CBF.

    More specifically, the CBF controller looks for a V such that

    V(safe) < 0
    V(unsafe) > 0
    dV/dt <= -lambda V

    This proves forward invariance of the 0-sublevel set of V, and since the safe set is
    a subset of this sublevel set, we prove that the unsafe region is not reachable from
    the safe region.
    """

    def __init__(
        self,
        dynamics_model: ControlAffineSystem,
        scenarios: ScenarioList,
        datamodule: EpisodicDataModule,
        experiment_suite: ExperimentSuite,
        h_hidden_layers: int = 2,
        h_hidden_size: int = 48,
        h_alpha: float = 0.9,
        V_hidden_layers: int = 2,
        V_hidden_size: int = 48,
        V_lambda: float = 0.0,
        cbf_relaxation_penalty: float = 50.0,
        controller_period: float = 0.01,
        primal_learning_rate: float = 1e-3,
        scale_parameter: float = 10.0,
        learn_shape_epochs: int = 0,
    ):
        """Initialize the controller.

        args:
            dynamics_model: the control-affine dynamics of the underlying system
            scenarios: a list of parameter scenarios to train on
            experiment_suite: defines the experiments to run during training
            cbf_hidden_layers: number of hidden layers to use for the CLBF network
            cbf_hidden_size: number of neurons per hidden layer in the CLBF network
            cbf_lambda: convergence rate for the CLBF
            cbf_relaxation_penalty: the penalty for relaxing CLBF conditions.
            controller_period: the timestep to use in simulating forward Vdot
            primal_learning_rate: the learning rate for SGD for the network weights,
                                  applied to the CLBF decrease loss
            scale_parameter: normalize non-angle data points to between +/- this value.
            learn_shape_epochs: number of epochs to spend just learning the shape
        """
        super(NeuralObsCBFController, self).__init__(
            dynamics_model=dynamics_model,
            scenarios=scenarios,
            experiment_suite=experiment_suite,
            cbf_lambda=h_alpha,
            cbf_relaxation_penalty=cbf_relaxation_penalty,
            controller_period=controller_period,
        )
        # Save the provided model
        # self.dynamics_model = dynamics_model
        self.scenarios = scenarios
        self.n_scenarios = len(scenarios)

        # Save the datamodule
        self.datamodule = datamodule

        # Save the experiments suits
        self.experiment_suite = experiment_suite

        # Save the other parameters
        self.primal_learning_rate = primal_learning_rate
        self.learn_shape_epochs = learn_shape_epochs

        # Compute and save the center and range of the state variables
        x_max, x_min = dynamics_model.state_limits
        self.x_center = (x_max + x_min) / 2.0
        self.x_range = (x_max - x_min) / 2.0
        # Scale to get the input between (-k, k), centered at 0
        self.k = scale_parameter
        self.x_range = self.x_range / self.k
        # We shouldn't scale or offset any angle dimensions
        self.x_center[self.dynamics_model.angle_dims] = 0.0
        self.x_range[self.dynamics_model.angle_dims] = 1.0

        # Some of the dimensions might represent angles. We want to replace these
        # dimensions with two dimensions: sin and cos of the angle. To do this, we need
        # to figure out how many numbers are in the expanded state
        n_angles = len(self.dynamics_model.angle_dims)

        self.n_dims_extended = self.dynamics_model.n_dims + self.dynamics_model.o_dims

        # Define the CLBF network, which we denote V
        assert h_alpha > 0
        assert h_alpha <= 1
        self.h_alpha = h_alpha
        assert V_lambda >= 0
        assert V_lambda <= 1
        self.V_lambda = V_lambda

        # ----------------------------------------------------------------------------
        # Define the BF network, which we denote h
        # ----------------------------------------------------------------------------
        self.h_hidden_layers = h_hidden_layers
        self.h_hidden_size = h_hidden_size
        num_h_inputs = self.dynamics_model.n_dims + self.dynamics_model.o_dims
        # We're going to build the network up layer by layer, starting with the input
        self.h_layers: OrderedDict[str, nn.Module] = OrderedDict()
        self.h_layers["input_linear"] = nn.Linear(num_h_inputs, self.h_hidden_size)
        self.h_layers["input_activation"] = nn.ReLU()
        for i in range(self.h_hidden_layers):
            self.h_layers[f"layer_{i}_linear"] = nn.Linear(
                self.h_hidden_size, self.h_hidden_size
            )
            self.h_layers[f"layer_{i}_activation"] = nn.ReLU()
        self.h_layers["output_linear"] = nn.Linear(self.h_hidden_size, 1)
        self.h_nn = nn.Sequential(self.h_layers)

        # ----------------------------------------------------------------------------
        # Define the LF network, which we denote V
        # ----------------------------------------------------------------------------
        self.V_hidden_layers = V_hidden_layers
        self.V_hidden_size = V_hidden_size
        # For turtlebot, the inputs to V are range and heading to the origin
        num_V_inputs = self.dynamics_model.n_dims
        # We're going to build the network up layer by layer, starting with the input
        self.V_layers: OrderedDict[str, nn.Module] = OrderedDict()
        self.V_layers["input_linear"] = nn.Linear(num_V_inputs, self.V_hidden_size)
        self.V_layers["input_activation"] = nn.ReLU()
        for i in range(self.V_hidden_layers):
            self.V_layers[f"layer_{i}_linear"] = nn.Linear(
                self.V_hidden_size, self.V_hidden_size
            )
            if i < self.V_hidden_layers - 1:
                self.V_layers[f"layer_{i}_activation"] = nn.ReLU()
        # Use the positive definite trick to encode V
        # self.V_layers["output_linear"] = nn.Linear(self.V_hidden_size, 1)
        self.V_nn = nn.Sequential(self.V_layers)

    # @property
    # def cbf_lambda(self):
    #     """Rename clf lambda to cbf"""
    #     return self.clf_lambda

    def prepare_data(self):
        return self.datamodule.prepare_data()

    def setup(self, stage: Optional[str] = None):
        return self.datamodule.setup(stage)

    def train_dataloader(self):
        return self.datamodule.train_dataloader()

    def val_dataloader(self):
        return self.datamodule.val_dataloader()

    def test_dataloader(self):
        return self.datamodule.test_dataloader()

    def V_with_jacobian(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the CLBF value and its Jacobian

        args:
            x: bs x self.dynamics_model.n_dims the points at which to evaluate the CLBF
        returns:
            V: bs tensor of CLBF values
            JV: bs x 1 x self.dynamics_model.n_dims Jacobian of each row of V wrt x
        """
        x_norm = x[:, :self.n_dims_extended]
        bs = x.shape[0]
        JV = torch.eye(self.n_dims_extended).repeat([bs, 1, 1]).type_as(x)

        # Now step through each layer in V
        V = x_norm
        for layer in self.h_nn:
            V = layer(V)

            if isinstance(layer, nn.Linear):
                JV = torch.matmul(layer.weight, JV)
            elif isinstance(layer, nn.Tanh):
                JV = torch.matmul(torch.diag_embed(1 - V ** 2), JV)
            elif isinstance(layer, nn.ReLU):
                JV = torch.matmul(torch.diag_embed(torch.sign(V)), JV)

        dodx = torch.hstack((x[:, self.n_dims_extended:], torch.zeros(bs, self.dynamics_model.q_dims))).type_as(x)
        dodx = dodx.reshape(bs, 1, self.dynamics_model.n_dims)
        J = JV[:, :, :self.dynamics_model.n_dims] + torch.bmm(JV[:, :, self.dynamics_model.n_dims:], dodx)

        return V, J

    def forward(self, x):
        """Determine the control input for a given state using a QP

        args:
            x: bs x self.dynamics_model.n_dims tensor of state
        returns:
            u: bs x self.dynamics_model.n_controls tensor of control inputs
        """
        return self.u(x[:, :self.n_dims_extended])

    def boundary_loss(
        self,
        x: torch.Tensor,
        goal_mask: torch.Tensor,
        safe_mask: torch.Tensor,
        unsafe_mask: torch.Tensor,
        accuracy: bool = False,
    ) -> List[Tuple[str, torch.Tensor]]:
        """
        Evaluate the loss on the CLBF due to boundary conditions

        args:
            x: the points at which to evaluate the loss,
            goal_mask: the points in x marked as part of the goal
            safe_mask: the points in x marked safe
            unsafe_mask: the points in x marked unsafe
            accuracy: if True, return the accuracy (from 0 to 1) as well as the losses
        returns:
            loss: a list of tuples containing ("category_name", loss_value).
        """
        eps = 1e-2
        # Compute loss to encourage satisfaction of the following conditions...
        loss = []

        h = self.h_nn(x[:, :self.n_dims_extended])

        #   2.) h < 0 in the safe region
        h_safe = h[safe_mask]
        safe_violation = F.relu(eps + h_safe)
        safe_h_term = 1e2 * safe_violation.mean()
        loss.append(("BF safe region term", safe_h_term))
        if accuracy:
            safe_h_acc = (safe_violation <= eps).sum() / safe_violation.nelement()
            loss.append(("BF safe region accuracy", safe_h_acc))

        #   3.) h > 0 in the unsafe region
        h_unsafe = h[unsafe_mask]
        unsafe_violation = F.relu(eps - h_unsafe)
        unsafe_h_term = 1e2 * unsafe_violation.mean()
        loss.append(("BF unsafe region term", unsafe_h_term))
        if accuracy:
            unsafe_h_acc = (unsafe_violation <= eps).sum() / unsafe_violation.nelement()
            loss.append(("BF unsafe region accuracy", unsafe_h_acc))

        return loss

    def descent_loss(
        self,
        x: torch.Tensor,
        goal_mask: torch.Tensor,
        safe_mask: torch.Tensor,
        unsafe_mask: torch.Tensor,
        accuracy: bool = False,
        requires_grad: bool = False,
    ) -> List[Tuple[str, torch.Tensor]]:
        """
        Evaluate the loss on the CLBF due to the descent condition

        args:
            x: the points at which to evaluate the loss,
            goal_mask: the points in x marked as part of the goal
            safe_mask: the points in x marked safe
            unsafe_mask: the points in x marked unsafe
            accuracy: if True, return the accuracy (from 0 to 1) as well as the losses
        returns:
            loss: a list of tuples containing ("category_name", loss_value).
        """
        # Compute loss to encourage satisfaction of the following conditions...
        loss = []

        # We'll encourage satisfying the CBF conditions by...
        #
        #   1) Minimize the relaxation needed to make the QP feasible.

        # Get the control input and relaxation from solving the QP, and aggregate
        # the relaxation across scenarios
        u_qp, qp_relaxation = self.solve_CLF_QP(x, requires_grad=requires_grad)
        qp_relaxation = torch.mean(qp_relaxation, dim=-1)

        # Minimize the qp relaxation to encourage satisfying the decrease condition
        qp_relaxation_loss = qp_relaxation.mean()
        loss.append(("QP relaxation", qp_relaxation_loss))

        return loss

    def training_step(self, batch, batch_idx):
        """Conduct the training step for the given batch"""
        # Extract the input and masks from the batch
        x, goal_mask, safe_mask, unsafe_mask = batch

        # Compute the losses
        component_losses = {}
        component_losses.update(
            self.boundary_loss(x, goal_mask, safe_mask, unsafe_mask)
        )
        if self.current_epoch > self.learn_shape_epochs:
            component_losses.update(
                self.descent_loss(
                    x, goal_mask, safe_mask, unsafe_mask, requires_grad=True
                )
            )

        # Compute the overall loss by summing up the individual losses
        total_loss = torch.tensor(0.0).type_as(x)
        # For the objectives, we can just sum them
        for _, loss_value in component_losses.items():
            if not torch.isnan(loss_value):
                total_loss += loss_value

        batch_dict = {"loss": total_loss, **component_losses}

        return batch_dict

    def training_epoch_end(self, outputs):
        """This function is called after every epoch is completed."""
        # Outputs contains a list for each optimizer, and we need to collect the losses
        # from all of them if there is a nested list
        if isinstance(outputs[0], list):
            outputs = itertools.chain(*outputs)

        # Gather up all of the losses for each component from all batches
        losses = {}
        for batch_output in outputs:
            for key in batch_output.keys():
                # if we've seen this key before, add this component loss to the list
                if key in losses:
                    losses[key].append(batch_output[key])
                else:
                    # otherwise, make a new list
                    losses[key] = [batch_output[key]]

        # Average all the losses
        avg_losses = {}
        for key in losses.keys():
            key_losses = torch.stack(losses[key])
            avg_losses[key] = torch.nansum(key_losses) / key_losses.shape[0]

        # Log the overall loss...
        self.log("Total loss / train", avg_losses["loss"], sync_dist=True)
        # And all component losses
        for loss_key in avg_losses.keys():
            # We already logged overall loss, so skip that here
            if loss_key == "loss":
                continue
            # Log the other losses
            self.log(loss_key + " / train", avg_losses[loss_key], sync_dist=True)

    def validation_step(self, batch, batch_idx):
        """Conduct the validation step for the given batch"""
        # Extract the input and masks from the batch
        x, goal_mask, safe_mask, unsafe_mask = batch

        # Get the various losses
        component_losses = {}
        component_losses.update(
            self.boundary_loss(x, goal_mask, safe_mask, unsafe_mask)
        )
        if self.current_epoch > self.learn_shape_epochs:
            component_losses.update(
                self.descent_loss(x, goal_mask, safe_mask, unsafe_mask)
            )

        # Compute the overall loss by summing up the individual losses
        total_loss = torch.tensor(0.0).type_as(x)
        # For the objectives, we can just sum them
        for _, loss_value in component_losses.items():
            if not torch.isnan(loss_value):
                total_loss += loss_value

        # Also compute the accuracy associated with each loss
        component_losses.update(
            self.boundary_loss(x, goal_mask, safe_mask, unsafe_mask, accuracy=True)
        )
        if self.current_epoch > self.learn_shape_epochs:
            component_losses.update(
                self.descent_loss(x, goal_mask, safe_mask, unsafe_mask, accuracy=True)
            )

        batch_dict = {"val_loss": total_loss, **component_losses}

        return batch_dict

    def validation_epoch_end(self, outputs):
        """This function is called after every epoch is completed."""
        # Gather up all of the losses for each component from all batches
        losses = {}
        for batch_output in outputs:
            for key in batch_output.keys():
                # if we've seen this key before, add this component loss to the list
                if key in losses:
                    losses[key].append(batch_output[key])
                else:
                    # otherwise, make a new list
                    losses[key] = [batch_output[key]]

        # Average all the losses
        avg_losses = {}
        for key in losses.keys():
            key_losses = torch.stack(losses[key])
            avg_losses[key] = torch.nansum(key_losses) / key_losses.shape[0]

        # Log the overall loss...
        self.log("Total loss / val", avg_losses["val_loss"], sync_dist=True)
        # And all component losses
        for loss_key in avg_losses.keys():
            # We already logged overall loss, so skip that here
            if loss_key == "val_loss":
                continue
            # Log the other losses
            self.log(loss_key + " / val", avg_losses[loss_key], sync_dist=True)

        # **Now entering spicetacular automation zone**
        # We automatically run experiments every few epochs

        # Only plot every 5 epochs
        if self.current_epoch % 5 != 0:
            return

        self.experiment_suite.run_all_and_log_plots(
            self, self.logger, self.current_epoch
        )

    @pl.core.decorators.auto_move_data
    def simulator_fn(
        self,
        x_init: torch.Tensor,
        num_steps: int,
        relaxation_penalty: Optional[float] = None,
    ):
        # Choose parameters randomly
        random_scenario = {}
        for param_name in self.scenarios[0].keys():
            param_max = max([s[param_name] for s in self.scenarios])
            param_min = min([s[param_name] for s in self.scenarios])
            random_scenario[param_name] = random.uniform(param_min, param_max)

        return self.dynamics_model.simulate(
            x_init,
            num_steps,
            self.u,
            guard=self.dynamics_model.out_of_bounds_mask,
            controller_period=self.controller_period,
            params=random_scenario,
        )

    def configure_optimizers(self):
        clbf_params = list(self.h_nn.parameters())

        clbf_opt = torch.optim.SGD(
            clbf_params,
            lr=self.primal_learning_rate,
            weight_decay=1e-6,
        )

        self.opt_idx_dict = {0: "clbf"}

        return [clbf_opt]
