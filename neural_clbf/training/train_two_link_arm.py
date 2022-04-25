from argparse import ArgumentParser
from importlib_metadata import requires

import torch
import torch.multiprocessing
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import numpy as np

from neural_clbf.controllers import NeuralObsCBFController, NeuralBFController
from neural_clbf.datamodules.episodic_datamodule import (
    EpisodicDataModule,
)
from neural_clbf.systems import TwoLinkArm2D
from neural_clbf.experiments import (
    ExperimentSuite,
    CLFContourExperiment,
    RolloutStateSpaceExperiment,
)
from neural_clbf.training.utils import current_git_hash


torch.multiprocessing.set_sharing_strategy("file_system")

batch_size = 64
controller_period = 0.01

start_x = torch.tensor(
    [
        [0, 0, 0, 0, 0.065, 0.275, 0.075],
        [0, np.pi/2, 0, 0, 0.06964194, 0.18307933, 0],
    ]
)
simulation_dt = 0.01


def main(args):
    # Define the scenarios
    nominal_params = {"m1": 5.76}
    scenarios = [
        nominal_params,
    ]

    # Define the dynamics model
    dynamics_model = TwoLinkArm2D(
        nominal_params,
        dt=simulation_dt,
        controller_dt=controller_period,
    )

    # Initialize the DataModule
    initial_conditions =[(-2.9671, 2.9671),
                         (-2.9671, 2.9671),
                         (-1.4385, 1.4385),
                         (-1.4385, 1.4385)]
    data_module = EpisodicDataModule(
        dynamics_model,
        initial_conditions,
        trajectories_per_episode=2,
        trajectory_length=150,
        fixed_samples=5000,
        max_points=20000,
        val_split=0.1,
        batch_size=64,
        # quotas={"safe": 0.4, "unsafe": 0.4, "goal": 0.2},
    )
    
    default_state = torch.tensor(np.concatenate((dynamics_model.goal_state, dynamics_model.get_observation_with_state(dynamics_model.goal_state))), requires_grad=True)

    # Define the experiment suite
    V_contour_experiment = CLFContourExperiment(
        "V_Contour",
        domain=[(-2.96705973, 2.96705972839), (-2.96705973, 2.96705972839)],
        n_grid=30,
        x_axis_index=0,
        y_axis_index=1,
        x_axis_label="$\\theta_1$",
        y_axis_label="$\\theta_2$",
        default_state=default_state,
        plot_unsafe_region=False,
    )
    rollout_experiment = RolloutStateSpaceExperiment(
        "Rollout",
        start_x,
        0,
        "$\\theta_1$",
        1,
        "$\\theta_2$",
        scenarios=scenarios,
        n_sims_per_start=1,
        t_sim=5.0,
    )
    experiment_suite = ExperimentSuite([V_contour_experiment, rollout_experiment])

    clbf_controller = NeuralObsCBFController(
        dynamics_model,
        scenarios,
        data_module,
        experiment_suite=experiment_suite,
        h_hidden_layers=2,
        h_hidden_size=48,
        h_alpha=0.3,
        controller_period=controller_period,
        learn_shape_epochs=100
    )

    # Initialize the logger and trainer
    tb_logger = pl_loggers.TensorBoardLogger(
        "logs/inverted_pendulum",
        name=f"commit_{current_git_hash()}",
    )
    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=tb_logger,
        reload_dataloaders_every_epoch=True,
        max_epochs=400,
    )

    # Train
    torch.autograd.set_detect_anomaly(True)
    trainer.fit(clbf_controller)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)
