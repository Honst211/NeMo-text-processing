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

from nemo_text_processing.text_normalization.ja.graph_utils import GraphFst
from nemo_text_processing.text_normalization.ja.utils import get_abs_path


class TelephoneFst(GraphFst):
    """
    Finite state transducer for classifying Japanese telephone numbers.
    
    Japanese telephone numbers typically follow these patterns:
    - Fixed line: 03-1234-5678, 06-1234-5678 (area code-local number)
    - Mobile: 090-1234-5678, 080-1234-5678, 070-1234-5678
    - Toll-free: 0120-123-456, 0800-123-456
    - With country code: +81-3-1234-5678, +81-90-1234-5678
    
    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="telephone", kind="classify", deterministic=deterministic)

        # TTS-friendly Japanese digit mapping for telephone numbers
        # Using katakana pronunciation for better TTS synthesis
        tts_digit_map = {
            "0": "ゼロ",
            "1": "イチ", 
            "2": "ニー",
            "3": "サン",
            "4": "ヨン",
            "5": "ゴ",
            "6": "ロク",
            "7": "ナナ",
            "8": "ハチ",
            "9": "キュー"
        }
        
        digit = pynini.union(*[pynini.cross(d, reading) for d, reading in tts_digit_map.items()])

        # TTS-friendly separators
        digit_separator = pynutil.insert("・")  # Nakaten (middle dot) for digit separation
        segment_separator = pynutil.insert("、")  # Japanese comma for segment separation
        
        # Telephone prompts in Japanese
        telephone_prompts = pynini.string_file(get_abs_path("data/telephone/telephone_prompt.tsv"))
        
        # International country code handling with TTS-friendly readings
        # Using more natural TTS pronunciations
        country_codes = pynini.union(
            pynini.cross("+81", "プラスハチイチ"),    # Japan  
            pynini.cross("+1", "プラスイチ"),        # US/Canada  
            pynini.cross("+44", "プラスヨンヨン"),    # UK
            pynini.cross("+49", "プラスヨンキュー"),  # Germany
            pynini.cross("+33", "プラスサンサン"),    # France
            pynini.cross("+39", "プラスサンキュー"),  # Italy
            pynini.cross("+86", "プラスハチロク"),    # China
            pynini.cross("+82", "プラスハチニー"),    # South Korea
            pynini.cross("+91", "プラスキューイチ"),  # India
            pynini.cross("+61", "プラスロクイチ"),    # Australia
            # Generic pattern for other countries: +XXX
            pynini.cross("+", "プラス") + digit + pynini.closure(digit, 0, 2)
        )
        
        country_code = (
            pynini.closure(telephone_prompts + pynutil.delete(" "), 0, 1)
            + country_codes
        )
        country_code = pynutil.insert("country_code: \"") + country_code + pynutil.insert("\"")
        country_code = country_code + pynini.closure(pynutil.delete("-"), 0, 1) + pynutil.insert(" ")

        # Comprehensive Japanese telephone number patterns based on official specifications
        # Reference: https://akinov.hatenablog.com/entry/2017/05/31/194421
        
        # Note: digit_with_space removed, using digit_separator instead
        
        # Variable length digit groups with TTS-friendly separators
        one_digit = digit
        two_digits = digit + digit_separator + digit
        three_digits = digit + digit_separator + digit + digit_separator + digit
        four_digits = digit + digit_separator + digit + digit_separator + digit + digit_separator + digit
        
        # 1. Mobile and PHS patterns: 0[5789]0-XXXX-XXXX (11 digits total)
        mobile_patterns = pynini.union(
            # 050-XXXX-XXXX (IP phone)
            pynini.cross("0", "ゼロ") + pynini.cross("5", "ゴ") + pynini.cross("0", "ゼロ") + pynutil.delete("-") + segment_separator + 
            four_digits + pynutil.delete("-") + segment_separator + four_digits,
            
            # 070-XXXX-XXXX (PHS, mobile)
            pynini.cross("0", "ゼロ") + pynini.cross("7", "ナナ") + pynini.cross("0", "ゼロ") + pynutil.delete("-") + segment_separator + 
            four_digits + pynutil.delete("-") + segment_separator + four_digits,
            
            # 080-XXXX-XXXX (mobile)
            pynini.cross("0", "ゼロ") + pynini.cross("8", "ハチ") + pynini.cross("0", "ゼロ") + pynutil.delete("-") + segment_separator + 
            four_digits + pynutil.delete("-") + segment_separator + four_digits,
            
            # 090-XXXX-XXXX (mobile)
            pynini.cross("0", "ゼロ") + pynini.cross("9", "キュー") + pynini.cross("0", "ゼロ") + pynutil.delete("-") + segment_separator + 
            four_digits + pynutil.delete("-") + segment_separator + four_digits
        )
        
        # 2. Fixed line patterns with area codes (0 + area_code + local_number = 10 digits)
        # Area code (1-4 digits) + Local code (1-4 digits) = 5 digits total, then + 4 digits
        
        # Flexible separators for phone number formatting
        separator = pynini.union(
            pynutil.delete("-"),     # dash
            pynutil.delete(" "),     # space  
            pynutil.delete("."),     # dot
        )
        
        # Parentheses pattern for local number: (XXXX)XXXX
        paren_pattern = pynutil.delete("(") + segment_separator + four_digits + pynutil.delete(")") + four_digits
        
        # Standard pattern for local number: XXXX{sep}XXXX  
        standard_pattern = four_digits + separator + segment_separator + four_digits
        
        # 2-1. 1-digit area code patterns: 0X{sep}XXXX{sep}XXXX or 0X(XXXX)XXXX
        one_digit_area = pynini.union(
            # 03 formats (Tokyo)
            pynini.cross("0", "ゼロ") + pynini.cross("3", "サン") + separator + (standard_pattern | paren_pattern),
            
            # 04 formats 
            pynini.cross("0", "ゼロ") + pynini.cross("4", "ヨン") + separator + (standard_pattern | paren_pattern),
            
            # 06 formats (Osaka)
            pynini.cross("0", "ゼロ") + pynini.cross("6", "ロク") + separator + (standard_pattern | paren_pattern)
        )
        
        # 2-2. 3-digit area code patterns: 0XX-XXX-XXXX or 0XX(XXX)XXXX
        two_digit_area = pynini.union(
            # 011-XXX-XXXX (Sapporo)
            pynini.cross("0", "ゼロ") + pynini.cross("1", "イチ") + pynini.cross("1", "イチ") + pynutil.delete("-") + segment_separator +
            three_digits + pynutil.delete("-") + segment_separator + four_digits,
            
            # 011(XXX)XXXX (Sapporo with parentheses)
            pynini.cross("0", "ゼロ") + pynini.cross("1", "イチ") + pynini.cross("1", "イチ") + pynutil.delete("(") + segment_separator +
            three_digits + pynutil.delete(")") + four_digits,
            
            # 052-XXX-XXXX (Nagoya)
            pynini.cross("0", "ゼロ") + pynini.cross("5", "ゴ") + pynini.cross("2", "ニー") + pynutil.delete("-") + segment_separator +
            three_digits + pynutil.delete("-") + segment_separator + four_digits,
            
            # 052(XXX)XXXX (Nagoya with parentheses)
            pynini.cross("0", "ゼロ") + pynini.cross("5", "ゴ") + pynini.cross("2", "ニー") + pynutil.delete("(") + segment_separator +
            three_digits + pynutil.delete(")") + four_digits,
            
            # 075-XXX-XXXX (Kyoto)
            pynini.cross("0", "ゼロ") + pynini.cross("7", "ナナ") + pynini.cross("5", "ゴ") + pynutil.delete("-") + segment_separator +
            three_digits + pynutil.delete("-") + segment_separator + four_digits,
            
            # 075(XXX)XXXX (Kyoto with parentheses)
            pynini.cross("0", "ゼロ") + pynini.cross("7", "ナナ") + pynini.cross("5", "ゴ") + pynutil.delete("(") + segment_separator +
            three_digits + pynutil.delete(")") + four_digits,
            
            # 092-XXX-XXXX (Fukuoka)
            pynini.cross("0", "ゼロ") + pynini.cross("9", "キュー") + pynini.cross("2", "ニー") + pynutil.delete("-") + segment_separator +
            three_digits + pynutil.delete("-") + segment_separator + four_digits,
            
            # 092(XXX)XXXX (Fukuoka with parentheses)
            pynini.cross("0", "ゼロ") + pynini.cross("9", "キュー") + pynini.cross("2", "ニー") + pynutil.delete("(") + segment_separator +
            three_digits + pynutil.delete(")") + four_digits,
            
            # 022-XXX-XXXX (Sendai)
            pynini.cross("0", "ゼロ") + pynini.cross("2", "ニー") + pynini.cross("2", "ニー") + pynutil.delete("-") + segment_separator +
            three_digits + pynutil.delete("-") + segment_separator + four_digits,
            
            # 022(XXX)XXXX (Sendai with parentheses)
            pynini.cross("0", "ゼロ") + pynini.cross("2", "ニー") + pynini.cross("2", "ニー") + pynutil.delete("(") + segment_separator +
            three_digits + pynutil.delete(")") + four_digits
        )
        
        # 2-3. 3-digit area code patterns: 0XXX-XX-XXXX
        three_digit_area = pynini.cross("0", "ゼロ") + digit + digit + digit + pynutil.delete("-") + segment_separator + \
                          two_digits + pynutil.delete("-") + segment_separator + four_digits
        
        # 2-4. 4-digit area code patterns: 0XXXX-X-XXXX  
        four_digit_area = pynini.cross("0", "ゼロ") + digit + digit + digit + digit + pynutil.delete("-") + segment_separator + \
                         one_digit + pynutil.delete("-") + segment_separator + four_digits
        
        # Combine all fixed line patterns
        fixed_patterns = pynini.union(one_digit_area, two_digit_area, three_digit_area, four_digit_area)
        
        # 3. Local numbers without area code (2-9 start, 5-8 digits)
        # These are less common but should be supported
        non_zero_digit = pynini.union(*[pynini.cross(str(i), tts_digit_map[str(i)]) for i in range(2, 10)])
        
        local_patterns = pynini.union(
            # XXXX-XXXX format (8 digits total, 4-4 split)
            non_zero_digit + three_digits + pynutil.delete("-") + segment_separator + four_digits,
            
            # XXX-XXXX format (7 digits total, 3-4 split)
            non_zero_digit + two_digits + pynutil.delete("-") + segment_separator + four_digits,
            
            # XX-XXXX format (6 digits total, 2-4 split)
            non_zero_digit + one_digit + pynutil.delete("-") + segment_separator + four_digits,
            
            # XXXXX format (5 digits, no separator)
            non_zero_digit + four_digits
        )
        
        # 4. Free dial patterns
        free_dial_patterns = pynini.union(
            # 0120-XXX-XXX
            pynini.cross("0", "ゼロ") + pynini.cross("1", "イチ") + pynini.cross("2", "ニー") + pynini.cross("0", "ゼロ") + 
            pynutil.delete("-") + segment_separator + 
            three_digits + pynutil.delete("-") + segment_separator + three_digits,
            
            # 0800-XXX-XXX  
            pynini.cross("0", "ゼロ") + pynini.cross("8", "ハチ") + pynini.cross("0", "ゼロ") + pynini.cross("0", "ゼロ") + 
            pynutil.delete("-") + segment_separator + 
            three_digits + pynutil.delete("-") + segment_separator + three_digits,
            
            # 0570-XXX-XXX (Navidial)
            pynini.cross("0", "ゼロ") + pynini.cross("5", "ゴ") + pynini.cross("7", "ナナ") + pynini.cross("0", "ゼロ") + 
            pynutil.delete("-") + segment_separator + 
            three_digits + pynutil.delete("-") + segment_separator + three_digits
        )
        
        # Combine all domestic patterns
        domestic_numbers = pynini.union(mobile_patterns, fixed_patterns, local_patterns, free_dial_patterns)
        number_part_domestic = pynutil.insert("number_part: \"") + domestic_numbers + pynutil.insert("\"")
        
        # International patterns (generic format for all countries)
        # Support various international phone number formats
        
        # Generic international number patterns
        # Format: X-XXX-XXXX or XX-XXXX-XXXX or XXX-XXX-XXXX etc.
        
        # 1-3 digit area/operator code
        intl_area_code = pynini.union(
            one_digit,                    # X
            two_digits,                   # XX  
            three_digits                  # XXX
        )
        
        # Number groups of various lengths (3-4 digits each)
        number_group = pynini.union(
            three_digits,                 # XXX
            four_digits                   # XXXX
        )
        
        # Flexible international phone number pattern
        # Supports: X-XXX-XXXX, XX-XXXX-XXXX, XXX-XXX-XXXX, XXX-XXXX-XXXX etc.
        international_numbers = pynini.union(
            # Pattern: Area-Number-Number  
            intl_area_code + separator + segment_separator + 
            number_group + separator + segment_separator + 
            number_group,
            
            # Pattern: Area-Number-Number-Number (for longer numbers)
            intl_area_code + separator + segment_separator + 
            number_group + separator + segment_separator + 
            number_group + separator + segment_separator + 
            number_group,
            
            # Pattern: Area-Number (for shorter numbers)
            intl_area_code + separator + segment_separator + 
            four_digits,
            
            # Japan specific patterns (without leading 0 for international)
            # 3-XXXX-XXXX (Tokyo), 6-XXXX-XXXX (Osaka)
            pynini.cross("3", "サン") + separator + segment_separator + standard_pattern,
            pynini.cross("6", "ロク") + separator + segment_separator + standard_pattern,
            
            # 50-XXXX-XXXX, 70-XXXX-XXXX, 80-XXXX-XXXX, 90-XXXX-XXXX (Japan mobile)
            pynini.union(
                pynini.cross("5", "ゴ"), pynini.cross("7", "ナナ"), 
                pynini.cross("8", "ハチ"), pynini.cross("9", "キュー")
            ) + pynini.cross("0", "ゼロ") + separator + segment_separator + standard_pattern
        )
        
        number_part_international = pynutil.insert("number_part: \"") + international_numbers + pynutil.insert("\"")

        # Extension handling (内線 - naisen in Japanese)
        extension = (
            pynutil.insert("extension: \"") + 
            pynini.closure(digit + pynutil.insert(" "), 0, 3) + 
            digit + 
            pynutil.insert("\"")
        )
        extension = pynini.closure(pynutil.insert(" ") + extension, 0, 1)

        # Complete telephone number patterns
        graph = pynini.union(
            # International numbers with country code
            country_code + number_part_international + extension,
            country_code + number_part_international,
            # Domestic numbers
            number_part_domestic + extension,
            number_part_domestic
        )

        final_graph = self.add_tokens(graph)
        self.fst = final_graph.optimize() 