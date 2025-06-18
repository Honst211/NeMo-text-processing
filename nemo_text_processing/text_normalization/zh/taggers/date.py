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

from nemo_text_processing.text_normalization.zh.graph_utils import GraphFst
from nemo_text_processing.text_normalization.zh.utils import get_abs_path


class DateFst(GraphFst):
    """
    Finite state transducer for classfying dates, e.g.
        2002年       -> tokens { date { year: "二零零二" } }
        2002-01-28   -> tokens { date { year: "二零零二" month: "一" day: "二十八"} }
        2002/01/28   -> tokens { date { year: "二零零二" month: "一" day: "二十八"} }
        2002.01.28   -> tokens { date { year: "二零零二" month: "一" day: "二十八"} }
        2002年2月    -> tokens { date { year: "二零零二" month: "二" } }
        2月11日      -> tokens { date { month: "二" day: "十一" } }
        2002/02      -> is an error format according to the national standard
        02/11        -> is an error format according to the national standard
        According to national standard, only when the year, month, and day are all exist, it is allowed to use symbols to separate them

    """

    def __init__(self, deterministic: bool = True, lm: bool = False):
        super().__init__(name="date", kind="classify", deterministic=deterministic)

        graph_digit = pynini.string_file(get_abs_path("data/number/digit.tsv"))
        graph_zero = pynini.string_file(get_abs_path("data/number/zero.tsv"))
        month = pynini.string_file(get_abs_path("data/date/months.tsv"))
        day = pynini.string_file(get_abs_path("data/date/day.tsv"))
        suffix = pynini.string_file(get_abs_path("data/date/suffixes.tsv"))

        delete_sign = pynutil.delete('/') | pynutil.delete('-') | pynutil.delete('.') | pynutil.delete('·')
        
        # 移除对"号"和"日"后缀的处理，这些情况将交给cardinal处理
        # 只保留不会与cardinal冲突的日期标识符

        # 严格限制有效日期范围：只有1-31的数字才被视为日期
        # 构建有效的日期数字范围（1-31）
        valid_day_numbers = pynini.union(*[str(i) for i in range(1, 32)])  # 1-31
        valid_day_numbers |= pynini.union(*[f"{i:02d}" for i in range(1, 32)])  # 01-31
        
        # 添加更严格的数字匹配：确保数字不会超出1-31范围
        # 明确排除32以上的数字，包括3位数
        invalid_day_numbers = pynini.union(*[str(i) for i in range(32, 1000)])  # 32-999
        # 排除所有3位数及以上（100+）
        three_digit_numbers = pynini.union(*[str(i) for i in range(100, 1000)])
        
        # 严格验证：只接受1-31范围内的数字
        valid_number_only = valid_day_numbers - invalid_day_numbers - three_digit_numbers
        
        # 严格的日期匹配：只在明确的日期上下文中识别日期
        # 移除了对"号"和"日"后缀的处理，这些将交给cardinal处理
        strict_day_match = valid_number_only @ day
        
        # grammar for only year, month, or day
        # atleast accep two digit to distinguish from year used for time
        # 不再单独处理带有"号"或"日"后缀的日期，这些交给cardinal处理
        only_year = (
            pynutil.insert("year: \"")
            + pynini.closure(graph_digit | graph_zero, 2)
            + pynutil.delete('年')
            + pynutil.insert("\"")
        )
        only_month = pynutil.insert("month: \"") + month + pynutil.delete('月') + pynutil.insert("\"")
        # 移除only_day的定义，不再单独处理日期
        # gh_1 - 只保留年和月的单独识别
        graph_only_date = only_year | only_month

        year_month = only_year + pynutil.insert(' ') + only_month
        # 移除包含only_day的组合，这些情况将交给cardinal处理
        # 现在只保留年月组合
        # gh_2
        graph_combination = year_month

        year_component = (
            pynutil.insert("year: \"")
            + pynini.closure(graph_digit | graph_zero, 2, 4)
            + delete_sign
            + pynutil.insert("\"")
        )
        month_component = pynutil.insert("month: \"") + month + delete_sign + pynutil.insert("\"")
        
        # 对于符号分隔的日期格式（如2024-01-15），保持完整的年月日格式
        # 这是唯一明确的日期标识格式，不会与cardinal冲突
        day_component_sign_separated = (
            pynutil.insert("day: \"") + strict_day_match + pynutil.insert("\"")
        )
        
        # gp_3 - 只处理完整的年-月-日格式
        graph_sign = year_component + pynutil.insert(' ') + month_component + pynutil.insert(' ') + day_component_sign_separated
        # gp_1+2+3
        graph_all = graph_only_date | graph_sign | graph_combination

        prefix = (
            pynini.accep('公元')
            | pynini.accep('西元')
            | pynini.accep('公元前')
            | pynini.accep('西元前')
            | pynini.accep('纪元')
            | pynini.accep('纪元前')
        )
        prefix_component = pynutil.insert("era: \"") + prefix + pynutil.insert("\"")
        # gp_prefix+(1,2,3) - 移除graph_ymd，只保留年月组合和单独年份
        graph_prefix = prefix_component + pynutil.insert(' ') + (year_month | only_year)

        suffix_component = pynutil.insert("era: \"") + suffix + pynutil.insert("\"")
        # gp_suffix +(1,2,3) - 移除graph_ymd，只保留年月组合和单独年份
        graph_suffix = (year_month | only_year) + pynutil.insert(' ') + suffix_component
        # gp_4
        graph_affix = graph_prefix | graph_suffix

        graph_suffix_year = (
            pynutil.insert("year: \"") + pynini.closure((graph_digit | graph_zero), 1) + pynutil.insert("\"")
        )
        graph_suffix_year = graph_suffix_year + pynutil.insert(' ') + suffix_component

        graph_with_era = graph_suffix_year | graph_affix

        graph = graph_only_date | graph_combination | graph_sign | graph_with_era

        # range
        symbol = pynini.accep("-") | pynini.accep("~") | pynini.accep("——") | pynini.accep("—")
        range_source = pynutil.insert("range: \"") + pynini.closure("从", 0, 1) + pynutil.insert("\"")
        range_goal = (
            pynutil.insert("range: \"")
            + (pynini.closure("到", 0, 1) | pynini.closure("至", 0, 1) | symbol)
            + pynutil.insert("\"")
        )
        graph_source = (
            range_source + pynutil.insert(' ') + graph + pynutil.insert(' ') + range_goal + pynutil.insert(' ') + graph
        )
        graph_goal = graph + pynutil.insert(' ') + range_goal + pynutil.insert(' ') + graph

        graph_range_final = graph_source | graph_goal

        # 移除内部高优先级权重，避免错误匹配超出范围的数字
        # 原代码: final_graph = pynutil.add_weight(graph, -2.0) | graph_range_final
        final_graph = graph | graph_range_final

        self.final_graph = final_graph.optimize()
        self.fst = self.add_tokens(self.final_graph).optimize()
