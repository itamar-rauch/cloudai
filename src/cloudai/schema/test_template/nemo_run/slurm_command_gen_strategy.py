# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Any, Dict, List, Union, cast

from cloudai import TestRun
from cloudai.systems.slurm.strategy import SlurmCommandGenStrategy
from cloudai.test_definitions.nemo_run import NeMoRunTestDefinition


class NeMoRunSlurmCommandGenStrategy(SlurmCommandGenStrategy):
    """Command generation strategy for NeMo 2.0 on Slurm systems."""

    def _parse_slurm_args(
        self, job_name_prefix: str, env_vars: Dict[str, str], cmd_args: Dict[str, Union[str, List[str]]], tr: TestRun
    ) -> Dict[str, Any]:
        base_args = super()._parse_slurm_args(job_name_prefix, env_vars, cmd_args, tr)

        tdef: NeMoRunTestDefinition = cast(NeMoRunTestDefinition, tr.test.test_definition)
        base_args.update({"image_path": tdef.docker_image.installed_path})

        if "target_experiments_dir" in cmd_args:
            base_args.update({"container_mounts": [f'{tr.output_path.absolute()}:{cmd_args["target_experiments_dir"]}']})

        return base_args

    def gen_srun_prefix(self, slurm_args: dict[str, Any], tr: TestRun) -> list[str]:
        tdef: NeMoRunTestDefinition = cast(NeMoRunTestDefinition, tr.test.test_definition)
        mounts = tdef.container_mounts(self.system.install_path)
        if slurm_args.get("container_mounts"):
            mounts.append(",".join(slurm_args["container_mounts"]))
        slurm_args["container_mounts"] = ",".join(mounts)
        cmd = super().gen_srun_prefix(slurm_args, tr)
        cmd.append(r'nsys profile -s none -o "/workspace/profile_${SLURM_JOB_ID}_node${SLURM_NODEID}_rank${SLURM_PROCID}.nsys-rep" -t cuda,nvtx --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop')
        return cmd
    
    def generate_test_command(
        self, env_vars: Dict[str, str], cmd_args: Dict[str, Union[str, List[str]]], tr: TestRun
    ) -> List[str]:
        tdef: NeMoRunTestDefinition = cast(NeMoRunTestDefinition, tr.test.test_definition)

        command = ["nemo", "llm", tdef.cmd_args.task, "--factory", tdef.cmd_args.recipe_name, "-y"]

        if tr.nodes:
            command.append(f"trainer.num_nodes={len(self.system.parse_nodes(tr.nodes))}")
        elif tr.num_nodes > 0:
            command.append(f"trainer.num_nodes={tr.num_nodes}")

        if tr.test.extra_cmd_args:
            command.append(tr.test.extra_cmd_args)

        return command
