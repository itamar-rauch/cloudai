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

from pathlib import Path
from typing import Optional

from cloudai import CmdArgs, TestDefinition
from cloudai.installer.installables import DockerImage, Installable


class NeMoRunCmdArgs(CmdArgs):
    """NeMoRun test command arguments."""

    docker_image_url: str
    task: str
    recipe_name: str
    target_experiments_dir: str
    nemo_fork_dir: str


class NeMoRunTestDefinition(TestDefinition):
    """Test object for NeMoLauncher."""

    cmd_args: NeMoRunCmdArgs
    _docker_image: Optional[DockerImage] = None

    @property
    def docker_image(self) -> DockerImage:
        if not self._docker_image:
            self._docker_image = DockerImage(url=self.cmd_args.docker_image_url)
        return self._docker_image
        
    def container_mounts(self, root: Path) -> list[str]:
        nemo_fork_dir = root / self.cmd_args.nemo_fork_dir
        return [
            f"{nemo_fork_dir.absolute()}:/opt/NeMo"
        ]
    
    @property
    def installables(self) -> list[Installable]:
        """Get list of installable objects."""
        return [self.docker_image]
