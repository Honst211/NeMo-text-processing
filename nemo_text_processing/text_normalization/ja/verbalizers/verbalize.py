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

from nemo_text_processing.text_normalization.ja.graph_utils import GraphFst, delete_space
from nemo_text_processing.text_normalization.ja.verbalizers.cardinal import CardinalFst
from nemo_text_processing.text_normalization.ja.verbalizers.date import DateFst
from nemo_text_processing.text_normalization.ja.verbalizers.decimal import DecimalFst
from nemo_text_processing.text_normalization.ja.verbalizers.fraction import FractionFst
from nemo_text_processing.text_normalization.ja.verbalizers.ordinal import OrdinalFst
from nemo_text_processing.text_normalization.ja.verbalizers.telephone import TelephoneFst
from nemo_text_processing.text_normalization.ja.verbalizers.time import TimeFst
from nemo_text_processing.text_normalization.ja.verbalizers.whitelist import WhiteListFst
from nemo_text_processing.text_normalization.ja.verbalizers.word import WordFst
from nemo_text_processing.text_normalization.ja.verbalizers.credit_card import CreditCardFst
from nemo_text_processing.text_normalization.ja.verbalizers.address_number import AddressNumberFst
from nemo_text_processing.text_normalization.ja.verbalizers.money import MoneyFst
from nemo_text_processing.text_normalization.ja.verbalizers.serial_number import SerialNumberFst


class VerbalizeFst(GraphFst):
    """
    Composes other verbalizer grammars.
    For deployment, this grammar will be compiled and exported to OpenFst Finate State Archiv (FAR) File.
    More details to deployment at NeMo/tools/text_processing_deployment.
    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple options (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="verbalize", kind="verbalize", deterministic=deterministic)

        date = DateFst(deterministic=deterministic)
        cardinal = CardinalFst(deterministic=deterministic)
        ordinal = OrdinalFst(deterministic=deterministic)
        decimal = DecimalFst(deterministic=deterministic)
        word = WordFst(deterministic=deterministic)
        fraction = FractionFst(deterministic=deterministic)
        telephone = TelephoneFst(deterministic=deterministic)
        credit_card = CreditCardFst(deterministic=deterministic)
        address_number = AddressNumberFst(deterministic=deterministic)
        time = TimeFst(deterministic=deterministic)
        whitelist = WhiteListFst(deterministic=deterministic)
        money = MoneyFst(deterministic=deterministic)
        serial_number = SerialNumberFst(deterministic=deterministic)

        graph = pynini.union(
            date.fst,
            cardinal.fst,
            ordinal.fst,
            decimal.fst,
            fraction.fst,
            telephone.fst,
            credit_card.fst,
            address_number.fst,
            word.fst,
            time.fst,
            whitelist.fst,
            money.fst,
            serial_number.fst
        )
        graph = pynini.closure(delete_space) + graph + pynini.closure(delete_space)

        self.fst = graph.optimize()
