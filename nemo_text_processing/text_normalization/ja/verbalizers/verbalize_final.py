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


import os
import re
import pynini
from pynini.lib import pynutil
from typing import Optional

from nemo_text_processing.text_normalization.ja.graph_utils import (
    NEMO_SIGMA,
    GraphFst,
    delete_space,
    generator_main
)
from nemo_text_processing.text_normalization.ja.verbalizers.verbalize import VerbalizeFst
from nemo_text_processing.text_normalization.ja.verbalizers.postprocessor import PostProcessor

# from nemo.utils import logging


class VerbalizeFinalFst(GraphFst):
    """
    Final class that composes all other verbalizer grammars.
    This class also converts the custom space marker <|space|> back to regular spaces.
    """

    def __init__(
        self,
        deterministic: bool = True,
        cache_dir: Optional[str] = None,
        overwrite_cache: bool = False
    ):
        super().__init__(name="verbalize_final", kind="verbalize", deterministic=deterministic)
        far_file = None
        if cache_dir is not None and cache_dir != "None":
            os.makedirs(cache_dir, exist_ok=True)
            far_file = os.path.join(cache_dir, f"jp_tn_{deterministic}_deterministic_verbalizer.far")
        if not overwrite_cache and far_file and os.path.exists(far_file):
            self.fst = pynini.Far(far_file, mode="r")["verbalize"]
        else:
            token_graph = VerbalizeFst(deterministic=deterministic)

            token_verbalizer = (
                pynutil.delete("tokens {") + delete_space + token_graph.fst + delete_space + pynutil.delete(" }")
            )
            verbalizer = pynini.closure(delete_space + token_verbalizer + delete_space)

            postprocessor = PostProcessor(
                remove_puncts=False,
                to_upper=False,
                to_lower=False,
                tag_oov=False,
            )

            # 添加将<|space|>转换回空格的规则
            space_marker_to_space = pynini.cdrewrite(
                pynini.cross("<|space|>", " "),
                "",
                "",
                NEMO_SIGMA
            ).optimize()

            # 组合所有规则
            self.fst = (verbalizer @ postprocessor.fst @ space_marker_to_space).optimize()

            if far_file:
                generator_main(far_file, {"verbalize": self.fst})

    def normalize(self, text: str) -> str:
        """
        Normalize the input text with additional post-processing.
        """
        # 首先使用基本的FST处理
        result = text
        try:
            result = pynini.compose(text, self.fst).string()
        except Exception:
            pass
        
        # 处理引号内的空格
        def restore_spaces(match):
            text = match.group(1)
            # 在连续的大写字母之间添加空格
            text = re.sub(r'([A-Z])([A-Z])', r'\1 \2', text)
            return f"『{text}』"
        
        # 使用正则表达式处理引号内的文本
        result = re.sub(r'『([^』]+)』', restore_spaces, result)
        
        # 确保所有<|space|>标记都被转换回空格
        result = result.replace("<|space|>", " ")
        
        return result
