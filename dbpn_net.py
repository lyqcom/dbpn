# 
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""dcgan"""
from mindspore import nn


class DBPN_NET(nn.Cell):
    """dcgan class"""
    def __init__(self, myTrainOneStepCellForD, myTrainOneStepCellForG):
        super(DBPN_NET, self).__init__(auto_prefix=True)
        self.myTrainOneStepCellForD = myTrainOneStepCellForD
        self.myTrainOneStepCellForG = myTrainOneStepCellForG

    def construct(self, input, target, target_ori, real_label, fake_label):
        output_D = self.myTrainOneStepCellForD(input, target, real_label, fake_label).view(-1)
        netD_loss = output_D.mean()
        output_G = self.myTrainOneStepCellForG(input, target, target_ori, real_label, fake_label).view(-1)
        netG_loss = output_G.mean()
        return netD_loss, netG_loss
