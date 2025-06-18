# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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


import pynini
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.zh.graph_utils import NEMO_NOT_QUOTE, GraphFst, delete_space


class ContextAwareColonFst(GraphFst):
    """
    上下文感知冒号处理的verbalizer
    处理时间格式和比例格式的输出
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="context_aware_colon", kind="verbalize", deterministic=deterministic)

        # 比例格式处理：ratio_first:ratio_second → 一比二
        ratio_first_part = (
            pynutil.delete("ratio_first:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE)
            + pynutil.delete("\"")
        )
        ratio_second_part = (
            pynutil.delete("ratio_second:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE)
            + pynutil.delete("\"")
        )
        
        # 比例格式输出：一比二
        graph_ratio = ratio_first_part + delete_space + pynutil.insert('比') + ratio_second_part

        # 时间格式处理：hours:minutes → 九点三十分
        hour_component = (
            pynutil.delete("hours:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE)
            + pynutil.delete("\"")
        )
        minute_component = (
            pynutil.delete("minutes:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE)
            + pynutil.delete("\"")
        )
        
        # 时间格式输出：九点三十分
        graph_time = hour_component + delete_space + minute_component

        # 合并两种格式
        final_graph = graph_ratio | graph_time

        # 删除tokens标记
        delete_tokens = self.delete_tokens(final_graph)
        self.fst = delete_tokens.optimize() 