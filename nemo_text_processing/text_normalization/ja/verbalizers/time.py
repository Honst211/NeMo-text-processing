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

from nemo_text_processing.text_normalization.ja.graph_utils import NEMO_NOT_QUOTE, GraphFst


class TimeFst(GraphFst):
    """
    Finite state transducer for verbalizing time e.g.
        time { hours: "三" minutes: "三十" } -> 三時三十分
        time { hours: "三" } -> 三時
        time { suffix: "今夜" hours: "零" } -> 今夜零時
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="time", kind="verbalize", deterministic=deterministic)

        hour_component = pynutil.delete("hours: \"") + pynini.closure(NEMO_NOT_QUOTE) + pynutil.delete("\"")
        minute_component = pynutil.delete("minutes: \"") + pynini.closure(NEMO_NOT_QUOTE) + pynutil.delete("\"")
        second_component = pynutil.delete("seconds: \"") + pynini.closure(NEMO_NOT_QUOTE) + pynutil.delete("\"")
        division_component = pynutil.delete("suffix: \"") + pynini.closure(NEMO_NOT_QUOTE) + pynutil.delete("\"")

        # Handle time components with proper formatting
        graph_basic_time = pynini.closure(division_component + pynutil.delete(" "), 0, 1) + (
            # Full time with hours, minutes, and seconds
            (hour_component + pynutil.delete(" ") + minute_component + pynutil.delete(" ") + second_component) |
            # Hours and minutes
            (hour_component + pynutil.delete(" ") + minute_component) |
            # Hours only (for cases where minutes are 0)
            hour_component |
            # Minutes only
            minute_component |
            # Seconds only
            second_component
        )

        final_graph = graph_basic_time

        delete_tokens = self.delete_tokens(final_graph)
        self.fst = delete_tokens.optimize()
