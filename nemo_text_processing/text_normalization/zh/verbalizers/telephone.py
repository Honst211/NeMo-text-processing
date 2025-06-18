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

from nemo_text_processing.text_normalization.zh.graph_utils import NEMO_NOT_QUOTE, GraphFst, delete_space, insert_space


class TelephoneFst(GraphFst):
    """
    Finite state transducer for verbalizing Chinese telephone numbers, e.g.
        telephone { country_code: "加八六" number_part: "一三八 二五六七 八九零一" extension: "三二一"  }
        -> 加八六，一三八、二五六七、八九零一，分机三二一

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="telephone", kind="verbalize", deterministic=deterministic)

        optional_country_code = pynini.closure(
            pynutil.delete("country_code: \"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
            + delete_space
            + pynutil.insert("，"),  # 中文逗号
            0,
            1,
        )

        number_part = (
            pynutil.delete("number_part: \"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynini.closure(pynutil.add_weight(pynutil.delete(" "), -0.0001), 0, 1)
            + pynutil.delete("\"")
        )

        optional_extension = pynini.closure(
            delete_space
            + pynutil.insert("，分机")  # "，分机" (extension in Chinese)
            + pynutil.delete("extension: \"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\""),
            0,
            1,
        )

        graph = optional_country_code + number_part + optional_extension
        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize() 