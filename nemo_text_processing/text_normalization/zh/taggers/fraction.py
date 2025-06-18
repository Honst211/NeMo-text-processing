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

from nemo_text_processing.text_normalization.zh.graph_utils import NEMO_CHAR, GraphFst
from nemo_text_processing.text_normalization.zh.utils import get_abs_path


class FractionFst(GraphFst):
    """
    Finite state transducer for classifying fraction, e.g.,
        1/2 -> tokens { fraction { denominator: "二" numerator: "一"} }
        5又1/2 -> tokens { fraction { integer_part: "五" denominator: "二" numerator: "一" } }
        5又2分之1 -> tokens { {} }
        2分之1 -> tokens { fraction { denominator: "二" numerator: "一"} }
        100分之1 -> tokens { fraction { denominator: "一百" numerator: "一"} }
        百分之1 -> tokens { fraction { denominator: "百" numerator: "一"} }
        98% -> tokens { fraction { denominator: "百" numerator: "九十八"} }

    Args:
        cardinal: CardinalFst, decimal: DecimalFst
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True, lm: bool = False):
        super().__init__(name="fraction", kind="classify", deterministic=deterministic)

        graph_cardinals = cardinal.just_cardinals
        graph_digit = pynini.string_file(get_abs_path("data/number/digit.tsv"))
        graph_zero = pynini.string_file(get_abs_path("data/number/zero.tsv"))

        slash = pynutil.delete('/')
        colon = pynutil.delete(':')  # 添加冒号分隔符支持比例格式
        morpheme = pynutil.delete('分之')
        suffix = pynini.union(
            "百",
            "千",
            "万",
            "十万",
            "百万",
            "千万",
            "亿",
            "十亿",
            "百亿",
            "千亿",
            "萬",
            "十萬",
            "百萬",
            "千萬",
            "億",
            "十億",
            "百億",
            "千億",
            "拾萬",
            "佰萬",
            "仟萬",
            "拾億",
            "佰億",
            "仟億",
            "拾万",
            "佰万",
            "仟万",
            "仟亿",
            "佰亿",
            "仟亿",
        )

        # 添加比例相关的上下文关键词
        ratio_context_keywords = pynini.union(
            "比例", "比率", "比值", "比重", "比", "占比", "配比", "倍率",
            "分数", "分值", "分比", "得分", "比分", "积分",
            "长宽", "长高", "宽高", "对比", "比较", "比赛"
        )

        # 构建上下文检测模式
        context_prefix = (
            ratio_context_keywords + 
            pynini.closure(pynini.union("：", ":", "是", "为", NEMO_CHAR), 0, 3)  # 可选连接词
        )
        context_suffix = ratio_context_keywords

        integer_component = pynutil.insert('integer_part: \"') + graph_cardinals + pynutil.insert("\"")
        denominator_component = pynutil.insert("denominator: \"") + graph_cardinals + pynutil.insert("\"")
        numerator_component = pynutil.insert("numerator: \"") + graph_cardinals + pynutil.insert("\"")

        graph_with_integer = (
            pynini.closure(integer_component + pynutil.delete('又'), 0, 1)
            + pynutil.insert(' ')
            + numerator_component
            + slash
            + pynutil.insert(' ')
            + denominator_component
        )  # 5又1/3

        graph_only_slash = numerator_component + slash + pynutil.insert(' ') + denominator_component
        
        # 按照用户4条规则实现精确的比例处理：
        # 只处理明确超出时间范围的数字，其他情况交给时间模块
        
        # 超出时间范围的数字：小时>24 或 分钟>59
        invalid_hour = pynini.union(*[str(i) for i in range(25, 1000)])  # 25以上 
        invalid_minute = pynini.union(*[str(i) for i in range(60, 1000)])  # 60以上
        any_number = pynini.union(*[str(i) for i in range(0, 1000)])
        
        # 构建明确的超出时间范围的比例格式
        # 第一个数字 >= 25 或 第二个数字 >= 60
        out_of_time_first = invalid_hour @ graph_cardinals  # 第一个数字超出小时范围
        out_of_time_second = invalid_minute @ graph_cardinals  # 第二个数字超出分钟范围
        any_cardinal = any_number @ graph_cardinals
        
        # 构建超出时间范围的比例格式
        ratio_first_component = pynutil.insert("ratio_first: \"") + any_cardinal + pynutil.insert("\"")
        ratio_second_component = pynutil.insert("ratio_second: \"") + any_cardinal + pynutil.insert("\"")
        
        ratio_first_out = pynutil.insert("ratio_first: \"") + out_of_time_first + pynutil.insert("\"")
        ratio_second_out = pynutil.insert("ratio_second: \"") + out_of_time_second + pynutil.insert("\"")
        
        # 构建基本的比例格式
        graph_ratio_basic = (
            ratio_first_component + colon + pynutil.insert(' ') + ratio_second_component
        )

        # 只有明确超出时间范围的才处理为比例
        graph_ratio_out_of_range = (
            (ratio_first_out + colon + pynutil.insert(' ') + ratio_second_component) |
            (ratio_first_component + colon + pynutil.insert(' ') + ratio_second_out)
        )
        
        # 带上下文的比例格式
        graph_ratio_with_context = (
            pynutil.insert("context_prefix: \"") + context_prefix +  # 保存前缀关键词
            pynutil.insert("\" ") +
            graph_ratio_basic +  # 数字部分
            pynini.closure(
                pynutil.insert(" context_suffix: \"") + context_suffix + pynutil.insert("\""),
                0, 1
            )  # 可选后缀关键词
        )
        
        # 合并所有比例格式，并设置权重
        graph_ratio_colon = pynini.union(
            pynutil.add_weight(graph_ratio_with_context, 0.1),  # 带上下文的比例优先级最高
            pynutil.add_weight(graph_ratio_out_of_range, 0.2)   # 超出时间范围的比例次之
        )

        graph_morpheme = (denominator_component + morpheme + pynutil.insert(' ') + numerator_component) | (
            integer_component
            + pynutil.delete('又')
            + pynutil.insert(' ')
            + denominator_component
            + morpheme
            + pynutil.insert(' ')
            + numerator_component
        )  # 5又3分之1

        graph_with_suffix = (
            pynini.closure(pynutil.insert("denominator: \"") + suffix + pynutil.insert("\""), 0, 1)
            + morpheme
            + pynutil.insert(' ')
            + numerator_component
        )  # 万分之1

        percentage = pynutil.delete('%')

        graph_decimal = (
            pynutil.insert('integer_part: \"')
            + pynini.closure(
                graph_cardinals
                + pynutil.delete('.')
                + pynutil.insert('点')
                + pynini.closure((graph_digit | graph_zero), 1)
            )
            + pynutil.insert("\"")
        )
        graph_decimal_percentage = pynini.closure(
            graph_decimal + percentage + pynutil.insert(' denominator: \"百"'), 1
        )  # 5.6%

        graph_integer_percentage = pynini.closure(
            (numerator_component) + percentage + pynutil.insert(' denominator: \"百"'), 1
        )  # 5%

        graph_hundred = pynutil.delete('100%') + pynutil.insert('numerator: \"百\" denominator: \"百"')
        # 100%

        graph_optional_sign = (pynini.closure(pynutil.insert("negative: ") + pynini.cross("-", "\"负\""))) | (
            pynutil.insert('negative: ')
            + pynutil.insert("\"")
            + (pynini.accep('负') | pynini.cross('負', '负'))
            + pynutil.insert("\"")
        )

        graph = pynini.union(
            graph_with_integer,
            graph_only_slash,
            graph_ratio_colon,  # 添加冒号比例格式
            graph_morpheme,
            graph_with_suffix,
            graph_decimal_percentage,
            graph_integer_percentage,
            graph_hundred,
        )
        graph_with_sign = (
            (graph_optional_sign + pynutil.insert(" ") + graph_with_integer)
            | (graph_optional_sign + pynutil.insert(" ") + graph_only_slash)
            | (graph_optional_sign + pynutil.insert(" ") + graph_ratio_colon)  # 添加冒号比例格式
            | (graph_optional_sign + pynutil.insert(" ") + graph_morpheme)
            | (graph_optional_sign + pynutil.insert(" ") + graph_with_suffix)
            | (graph_optional_sign + pynutil.insert(" ") + graph_integer_percentage)
            | (graph_optional_sign + pynutil.insert(" ") + graph_decimal_percentage)
            | (graph_optional_sign + pynutil.insert(" ") + graph_hundred)
        )

        final_graph = graph | pynutil.add_weight(graph_with_sign, -3.0)

        self.just_fractions = graph.optimize()
        self.fractions = final_graph.optimize()

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
