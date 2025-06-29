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

import pynini
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.ja.graph_utils import (
    NEMO_SIGMA,
    NEMO_CHAR,
    NEMO_SPACE,
    GraphFst,
    generator_main
)
from nemo_text_processing.text_normalization.ja.taggers.cardinal import CardinalFst
from nemo_text_processing.text_normalization.ja.taggers.date import DateFst
from nemo_text_processing.text_normalization.ja.taggers.decimal import DecimalFst
from nemo_text_processing.text_normalization.ja.taggers.fraction import FractionFst
from nemo_text_processing.text_normalization.ja.taggers.ordinal import OrdinalFst
from nemo_text_processing.text_normalization.ja.taggers.punctuation import PunctuationFst
from nemo_text_processing.text_normalization.ja.taggers.telephone import TelephoneFst
from nemo_text_processing.text_normalization.ja.taggers.time import TimeFst
from nemo_text_processing.text_normalization.ja.taggers.whitelist import WhiteListFst
from nemo_text_processing.text_normalization.ja.taggers.word import WordFst
from nemo_text_processing.text_normalization.ja.taggers.preprocessor import PreProcessorFst
from nemo_text_processing.text_normalization.ja.taggers.credit_card import CreditCardFst
from nemo_text_processing.text_normalization.ja.taggers.address_number import AddressNumberFst
from nemo_text_processing.text_normalization.ja.taggers.money import MoneyFst
from nemo_text_processing.text_normalization.ja.taggers.serial_number import SerialNumberFst


class ClassifyFst(GraphFst):
    """
    Final class that composes all other classification grammars. This class can process an entire sentence including punctuation.
    For deployment, this grammar will be compiled and exported to OpenFst Finate State Archiv (FAR) File.
    More details to deployment at NeMo/tools/text_processing_deployment.

    Args:
        input_case: accepting either "lower_cased" or "cased" input.
        deterministic: if True will provide a single transduction option,
            for False multiple options (used for audio-based normalization)
        cache_dir: path to a dir with .far grammar file. Set to None to avoid using cache.
        overwrite_cache: set to True to overwrite .far files
        whitelist: path to a file with whitelist replacements
    """

    def __init__(
        self,
        input_case: str,
        deterministic: bool = True,
        cache_dir: str = None,
        overwrite_cache: bool = False,
        whitelist: str = None,
    ):
        super().__init__(name="tokenize_and_classify", kind="classify", deterministic=deterministic)

        far_file = None
        if cache_dir is not None and cache_dir != "None":
            os.makedirs(cache_dir, exist_ok=True)
            whitelist_file = os.path.basename(whitelist) if whitelist else ""
            far_file = os.path.join(cache_dir, f"ja_tn_{deterministic}_deterministic_{whitelist_file}_tokenize.far")
        if not overwrite_cache and far_file and os.path.exists(far_file):
            self.fst = pynini.Far(far_file, mode="r")["tokenize_and_classify"]
        else:
            # Initialize preprocessor first
            preprocessor = PreProcessorFst(fullwidth_to_halfwidth=True)
            
            cardinal = CardinalFst(deterministic=deterministic)
            date = DateFst(cardinal=cardinal, deterministic=deterministic)
            decimal = DecimalFst(cardinal=cardinal, deterministic=deterministic)
            time = TimeFst(cardinal=cardinal, deterministic=deterministic)
            fraction = FractionFst(cardinal=cardinal, deterministic=deterministic)
            ordinal = OrdinalFst(cardinal=cardinal, deterministic=deterministic)
            telephone = TelephoneFst(deterministic=deterministic)
            credit_card = CreditCardFst(deterministic=deterministic)
            address_number = AddressNumberFst(deterministic=deterministic)
            serial_number = SerialNumberFst(deterministic=deterministic)
            whitelist = WhiteListFst(deterministic=deterministic)
            word = WordFst(deterministic=deterministic)
            punctuation = PunctuationFst(deterministic=deterministic)
            money = MoneyFst(cardinal=cardinal, deterministic=deterministic)

            classify = pynini.union(
                pynutil.add_weight(cardinal.fst, 0.9),     # Higher priority for numbers
                pynutil.add_weight(address_number.fst, 1.1),  # Same priority as telephone/credit card
                pynutil.add_weight(telephone.fst, 1.2),    # Lower priority for telephone 
                pynutil.add_weight(credit_card.fst, 1.2),  # Same priority as telephone
                pynutil.add_weight(money.fst, 0.8),        # Add money with medium priority
                pynutil.add_weight(serial_number.fst, 1.5),        # Add money with medium priority
                pynutil.add_weight(date.fst, 1.1),
                pynutil.add_weight(fraction.fst, 1.0),
                pynutil.add_weight(time.fst, 1.1),
                pynutil.add_weight(whitelist.fst, 1.1),
                pynutil.add_weight(decimal.fst, 3.05),
                pynutil.add_weight(ordinal.fst, 1.1),
                pynutil.add_weight(punctuation.fst, 1.0),
                pynutil.add_weight(word.fst, 100),
            )

            

            # 简化token构建方式
            token = pynutil.insert("tokens { ") + classify + pynutil.insert(" } ")
            
            # 使用更简单的方式构建tagger
            tagger = pynini.closure(
                token | pynini.cross(NEMO_SPACE, NEMO_SPACE),
                1
            )

            # 先应用preprocessor
            text = preprocessor.fst
            
            # 然后应用tagger
            self.fst = pynini.compose(text, tagger).optimize()

            if far_file:
                generator_main(far_file, {"tokenize_and_classify": self.fst})
