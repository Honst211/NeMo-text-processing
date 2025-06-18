# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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


class FractionFst(GraphFst):
    """
    Finite state transducer for verbalizing fraction e.g.
        tokens { fraction { denominator: "二" numerator: "一"} } -> 二分之一
        tokens { fraction { integer_part: "一" denominator: "二" numerator: "一" } } -> 一又二分之一
    """

    def __init__(self, decimal: GraphFst, deterministic: bool = True, lm: bool = False):
        super().__init__(name="fraction", kind="verbalize", deterministic=deterministic)

        graph_decimal = decimal.decimal_component

        integer_part = (
            pynutil.delete("integer_part:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE)
            + pynutil.insert("又")
            + pynutil.delete("\"")
        )
        denominator_part = (
            pynutil.delete("denominator:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE)
            + pynutil.delete("\"")
        )
        numerator_part = (
            pynutil.delete("numerator:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE)
            + pynutil.delete("\"")
        )
        sign_part = (
            pynutil.delete("positive:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE)
            + pynutil.delete("\"")
        ) | (
            pynutil.delete("negative:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE)
            + pynutil.delete("\"")
        )
        
        # 添加比例格式的处理组件
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

        # 添加上下文关键词处理
        context_prefix = (
            pynutil.delete("context_prefix: \"") + 
            pynini.closure(NEMO_NOT_QUOTE) + 
            pynutil.delete("\"")
        )
        context_suffix = (
            pynutil.delete("context_suffix: \"") + 
            pynini.closure(NEMO_NOT_QUOTE) + 
            pynutil.delete("\"")
        )

        graph_with_integer = (
            integer_part + delete_space + denominator_part + delete_space + pynutil.insert('分之') + numerator_part
        )
        graph_no_integer = denominator_part + delete_space + pynutil.insert('分之') + numerator_part
        
        # 比例格式：比例1:2 -> 比例一比二
        graph_ratio = (
            pynini.closure(context_prefix + delete_space, 0, 1) +  # 可选前缀
            ratio_first_part + delete_space + pynutil.insert('比') + ratio_second_part +
            pynini.closure(delete_space + context_suffix, 0, 1)    # 可选后缀
        )
        
        graph = graph_with_integer | graph_no_integer | graph_ratio

        graph_with_decimal = (
            denominator_part
            + delete_space
            + pynutil.insert('分之')
            + pynutil.delete("integer_part: \"")
            + pynini.closure(NEMO_NOT_QUOTE)
            + pynutil.delete("\"")
        )
        graph_with_sign = sign_part + delete_space + (graph | graph_with_decimal)

        # 添加带符号的比例格式支持
        graph_ratio_with_sign = sign_part + delete_space + graph_ratio

        final_graph = graph_with_sign | graph | graph_with_decimal | graph_ratio_with_sign
        self.fraction = final_graph

        delete_tokens = self.delete_tokens(final_graph)
        self.fst = delete_tokens.optimize()
