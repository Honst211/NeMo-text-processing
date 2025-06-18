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

from nemo_text_processing.text_normalization.zh.graph_utils import NEMO_DIGIT, NEMO_CHAR, NEMO_SPACE, GraphFst


class TelephoneFst(GraphFst):
    """
    Finite state transducer for classifying Chinese telephone numbers.
    
    支持的格式：
    - 固定电话：010-1234-5678, 021-8765-4321, 0755-1234567
    - 手机号码：138-1234-5678, 186-9876-5432
    - 客服电话：400-123-456, 800-987-654, 95588
    - 国际电话：+86-138-1234-5678
    - 上下文感知：检测"电话"、"热线"、"客服"等关键词提升短号码识别
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="telephone", kind="classify", deterministic=deterministic)

        # 中文数字映射 - 电话号码专用读音
        digit_to_chinese = {
            "0": "零", "1": "幺", "2": "二", "3": "三", "4": "四",
            "5": "五", "6": "六", "7": "七", "8": "八", "9": "九"
        }
        
        # 基础数字转换
        digit = pynini.union(*[pynini.cross(k, v) for k, v in digit_to_chinese.items()])
        
        # 分隔符处理：删除输入分隔符，插入中文分隔符
        separator_delete = pynini.union(
            pynini.cross("-", "杠"),    # 连字符转换为"杠"
            pynutil.delete(" "),    # 空格仍然删除
            pynutil.delete("."),    # 点号删除
            pynutil.delete("("),    # 左括号删除
            pynutil.delete(")"),    # 右括号删除
        )
        
        # 特殊的连字符处理：保留"杠"音
        dash_to_gang = pynini.cross("-", "杠")
        
        # 中文输出分隔符
        segment_sep = pynutil.insert("、")  # 段落分隔符
        digit_sep = pynutil.insert(" ")    # 数字间分隔符
        
        # ============================================
        # 新增：连续手机号码识别（无分隔符）
        # ============================================
        
        # 连续11位手机号码识别：1xxxxxxxxxx
        # 第一位必须是1，第二位是3-9
        continuous_mobile_11 = (
            pynini.cross("1", "幺") +
            pynini.union(*[pynini.cross(str(i), digit_to_chinese[str(i)]) for i in range(3, 10)]) +
            pynini.union(*[pynini.cross(str(i), digit_to_chinese[str(i)]) for i in range(0, 10)]) +
            segment_sep +  # 第一个分隔符：3位后
            # 接下来4位数字
            digit + digit + digit + digit + segment_sep +  # 第二个分隔符：7位后  
            # 最后4位数字
            digit + digit + digit + digit
        )
        
        # 连续10位客服号码识别：400xxxxxxx, 800xxxxxxx
        continuous_service_10 = pynini.union(
            pynini.cross("400", "四零零") + segment_sep +
            digit + digit + digit_sep + digit + digit + digit_sep + digit,
            pynini.cross("800", "八零零") + segment_sep +
            digit + digit + digit_sep + digit + digit + digit_sep + digit
        )
        
        # 连续5位银行客服：95xxx
        continuous_service_5 = (
            pynini.cross("95", "九五") + 
            digit + digit_sep + digit + digit_sep + digit
        )
        
        # ============================================
        # 1. 上下文关键词检测
        # ============================================
        
        # 电话相关的前缀词汇（在号码前出现）
        phone_prefix_keywords = pynini.union(
            "电话", "热线", "客服", "服务", "咨询", "联系", "预约", "订购",
            "拨打", "致电", "呼叫", "号码", "专线", "座机", "手机",
            "客服电话", "咨询热线", "服务热线", "预约电话", "联系电话",
            "销售热线", "投诉电话", "报修电话", "紧急电话", "急救电话",
            "拨"
        )
        
        # 电话相关的后缀词汇（在号码后出现）
        phone_suffix_keywords = pynini.union(
            "号码", "电话", "热线", "专线"
        )
        
        # 上下文检测：前缀模式 (关键词 + 可选分隔符 + 号码)
        # 支持：电话：xxx、电话号码：xxx、咨询热线xxx等
        context_prefix = (
            phone_prefix_keywords + 
            pynini.closure(pynini.union("：", ":", "是", "为", NEMO_SPACE), 0, 3)  # 可选连接词
        )
        
        # 上下文检测：后缀模式 (号码 + 关键词)
        # 支持：xxx号、xxx电话等
        context_suffix = phone_suffix_keywords
        
        # 宽松的上下文窗口：前后10个字符内检测关键词
        any_chars = pynini.closure(NEMO_CHAR, 0, 10)
        
        # ============================================
        # 2. 国际区号处理
        # ============================================
        country_codes = pynini.union(
            pynini.cross("+86", "加八六"),   # 中国
            pynini.cross("+1", "加一"),      # 美国/加拿大
            pynini.cross("+44", "加四四"),   # 英国
            pynini.cross("+81", "加八一"),   # 日本
            pynini.cross("+82", "加八二"),   # 韩国
        )
        
        # ============================================  
        # 3. 中国电话号码格式识别
        # ============================================
        
        # 3.1 固定电话区号（3-4位，以0开头）
        area_code_3digit = (
            pynini.cross("0", "零") +
            pynini.union(*[pynini.cross(str(i), digit_to_chinese[str(i)]) for i in range(1, 10)]) +
            pynini.union(*[pynini.cross(str(i), digit_to_chinese[str(i)]) for i in range(0, 10)])
        )
        
        area_code_4digit = (
            pynini.cross("0", "零") +
            pynini.union(*[pynini.cross(str(i), digit_to_chinese[str(i)]) for i in range(1, 10)]) +
            pynini.union(*[pynini.cross(str(i), digit_to_chinese[str(i)]) for i in range(0, 10)]) +
            pynini.union(*[pynini.cross(str(i), digit_to_chinese[str(i)]) for i in range(0, 10)])
        )
        
        # 3.2 手机号码前缀（1开头，第二位是3-9）
        mobile_prefix = (
            pynini.cross("1", "幺") +
            pynini.union(*[pynini.cross(str(i), digit_to_chinese[str(i)]) for i in range(3, 10)]) +
            pynini.union(*[pynini.cross(str(i), digit_to_chinese[str(i)]) for i in range(0, 10)])
        )
        
        # 3.3 客服电话前缀
        service_prefix_400_800 = pynini.union(
            pynini.cross("400", "四零零"),
            pynini.cross("800", "八零零")
        )
        
        # ============================================
        # 4. 数字组合构建器
        # ============================================
        
        # 修复：生成连续数字序列，避免数字组合（如"七八"被误读为"七十八"）
        def make_continuous_digits(length: int):
            """生成指定长度的连续数字序列，数字连续读出，防止组合但保持流畅"""
            if length == 1:
                return digit
            result = digit
            for i in range(1, length):
                # 直接连接数字，不添加分隔符，依靠上下文防止组合读音
                result = result + digit
            return result
        
        # 常用数字组合 - 使用连续数字生成器，配合适当的段落分隔
        digit_1 = make_continuous_digits(1)
        digit_2 = make_continuous_digits(2) 
        digit_3 = make_continuous_digits(3)
        digit_4 = make_continuous_digits(4)
        digit_5 = make_continuous_digits(5)
        digit_6 = make_continuous_digits(6)
        digit_7 = make_continuous_digits(7)
        digit_8 = make_continuous_digits(8)
        
        # ============================================
        # 5. 电话号码格式模式
        # ============================================
        
        # 5.1 固定电话格式（修复：标准中国固定电话只有一个分隔符）
        # 3位区号 + 8位号码：010-12345678 (不是010-1234-5678)
        fixed_3_8 = (
            area_code_3digit + pynutil.delete("-") + segment_sep +
            digit_8  # 完整的8位号码，不再分段
        )
        
        # 3位区号 + 7位号码：010-1234567  
        fixed_3_7 = (
            area_code_3digit + pynutil.delete("-") + segment_sep +
            digit_7  # 完整的7位号码，不再分段
        )
        
        # 4位区号 + 8位号码：0755-12345678 (不是0755-1234-5678)
        fixed_4_8 = (
            area_code_4digit + pynutil.delete("-") + segment_sep +
            digit_8  # 完整的8位号码，不再分段
        )
        
        # 4位区号 + 7位号码：0755-1234567 (不是0755-123-4567)
        fixed_4_7 = (
            area_code_4digit + pynutil.delete("-") + segment_sep +
            digit_7  # 完整的7位号码，不再分段
        )
        
        # 4位区号 + 6位号码：0755-123456
        fixed_4_6 = (
            area_code_4digit + pynutil.delete("-") + segment_sep +
            digit_6  # 完整的6位号码，不再分段
        )
        
        # 5.2 手机号码格式（修正分隔符处理）
        mobile_format = (
            mobile_prefix + pynutil.delete("-") + segment_sep +
            digit_4 + pynutil.delete("-") + segment_sep +
            digit_4
        )
        
        # 5.3 客服电话格式（修正分隔符处理）
        service_400_800 = (
            service_prefix_400_800 + pynutil.delete("-") + segment_sep +
            digit_3 + pynutil.delete("-") + segment_sep +
            digit_4
        )
        
        service_400_800_short = (
            service_prefix_400_800 + pynutil.delete("-") + segment_sep +
            digit_3 + pynutil.delete("-") + segment_sep +
            digit_3
        )
        
        # ============================================
        # 6. 上下文感知的短号码识别
        # ============================================
        
        # 6.1 紧急电话（需要上下文支持）
        emergency_numbers_basic = pynini.union(
            pynini.cross("110", "幺 幺 零"),    # 报警
            pynini.cross("119", "幺 幺 九"),    # 火警
            pynini.cross("120", "幺 二 零"),    # 急救
            pynini.cross("122", "幺 二 二"),    # 交通事故
            pynini.cross("114", "幺 幺 四"),    # 号码查询
        )
        
        # 客服电话（5位数字）
        service_numbers_basic = pynini.union(
            pynini.cross("10086", "幺 零 零 八 六"),  # 移动客服
            pynini.cross("10010", "幺 零 零 幺 零"),  # 联通客服  
            pynini.cross("10000", "幺 零 零 零 零"),  # 电信客服
            pynini.cross("95588", "九 五 五 八 八"),  # 银行客服
        )
        
        # 带上下文的紧急电话
        emergency_with_context = pynini.union(
            # 前缀上下文：拨打110、紧急电话119等
            context_prefix + emergency_numbers_basic,
            # 后缀上下文：110报警电话、119号等（直接连接，无中间字符）
            emergency_numbers_basic + context_suffix,
            # 直接匹配（权重较低）
            pynutil.add_weight(emergency_numbers_basic, 1.5)
        )
        
        # 带上下文的客服电话
        service_with_context = pynini.union(
            # 前缀上下文：电话10086、客服10010等
            context_prefix + service_numbers_basic,
            # 后缀上下文：10086客服、95588号码等
            service_numbers_basic + context_suffix,
            # 直接匹配（权重较低）
            pynutil.add_weight(service_numbers_basic, 1.5)
        )
        
        # ============================================
        # 7. 泛化匹配模式
        # ============================================
        
        # 检查是否以电话号码特征开头
        phone_starter = pynini.union(
            # 以0开头（固定电话）
            pynini.cross("0", "零") + digit + pynini.closure(digit, 1, 2),
            # 以1开头且第二位是3-9（手机号）  
            pynini.cross("1", "幺") + pynini.union(*[pynini.cross(str(i), digit_to_chinese[str(i)]) for i in range(3, 10)]),
            # 以4/8开头（客服电话）
            pynini.union(
                pynini.cross("4", "四"),
                pynini.cross("8", "八")
            ) + pynini.cross("0", "零") + pynini.cross("0", "零"),
        )
        
        # 泛化电话格式：确保有连字符分隔的数字序列
        generic_phone = (
            phone_starter + 
            pynini.closure(digit_sep + digit, 0, 2) +  # 完成第一段
            separator_delete + segment_sep +           # 第一个分隔符
            pynini.closure(digit + digit_sep, 2, 4) + digit +  # 中间段
            pynini.closure(                            # 可能的第三段
                separator_delete + segment_sep +
                pynini.closure(digit + digit_sep, 2, 4) + digit,
                0, 1
            )
        )
        
        # ============================================
        # 8. 国际电话格式
        # ============================================
        
        # 中国号码格式
        china_international = (
            pynini.cross("+86", "加八六") + separator_delete + pynutil.insert("，") +
            pynini.union(
                mobile_format,
                fixed_3_8,
                fixed_3_7,
                fixed_4_8,
                fixed_4_7,
                service_400_800
            )
        )
        
        # 修复：支持灵活的国际电话格式，而不是固定的3-3-4模式
        # 国际电话的通用特征：+国家码-地区码-号码
        # 不同国家有不同的分组方式，需要支持多种组合
        
        # 灵活的数字段：支持1-4位数字的各种组合
        flexible_digit_group = pynini.union(
            digit_1,    # 1位：如日本的"3"
            digit_2,    # 2位：如英国的"20"
            digit_3,    # 3位：如美国的"555"
            digit_4     # 4位：如"1234"
        )
        
        # 其他国家灵活格式：+CC-地区码-号码段1-号码段2
        # 支持多种实际的国际电话格式
        other_international_flexible = (
            pynini.union(
                pynini.cross("+1", "加一"),      # 美国/加拿大
                pynini.cross("+44", "加四四"),   # 英国
                pynini.cross("+81", "加八一"),   # 日本
                pynini.cross("+82", "加八二"),   # 韩国
                pynini.cross("+33", "加三三"),   # 法国
                pynini.cross("+49", "加四九"),   # 德国
                pynini.cross("+61", "加六一"),   # 澳大利亚
                pynini.cross("+91", "加九一"),   # 印度
            ) + separator_delete + pynutil.insert("，") +
            # 第一段：地区码（1-4位数字）
            flexible_digit_group + separator_delete + segment_sep +
            # 第二段：主号码第一部分（3-4位数字）
            pynini.union(digit_3, digit_4) + separator_delete + segment_sep +
            # 第三段：主号码第二部分（4位数字）
            digit_4
        )
        
        # 也支持更长的号码格式：+CC-X-XXX-XXX-XXXX
        other_international_extended = (
            pynini.union(
                pynini.cross("+1", "加一"),      # 美国/加拿大
                pynini.cross("+44", "加四四"),   # 英国
                pynini.cross("+81", "加八一"),   # 日本
            ) + separator_delete + pynutil.insert("，") +
            # 地区码（1-3位）
            pynini.union(digit_1, digit_2, digit_3) + separator_delete + segment_sep +
            # 三段号码：XXX-XXX-XXXX 或 XXX-XXXX
            pynini.union(
                # 格式1: XXX-XXX-XXXX
                digit_3 + separator_delete + segment_sep + digit_3 + separator_delete + segment_sep + digit_4,
                # 格式2: XXX-XXXX
                digit_3 + separator_delete + segment_sep + digit_4,
                # 格式3: XXXX-XXXX
                digit_4 + separator_delete + segment_sep + digit_4
            )
        )
        
        # 原始固定格式（保持向后兼容）
        other_international_fixed = (
            pynini.union(
                pynini.cross("+1", "加一"),      # 美国/加拿大
                pynini.cross("+44", "加四四"),   # 英国
                pynini.cross("+81", "加八一"),   # 日本
                pynini.cross("+82", "加八二"),   # 韩国
            ) + separator_delete + pynutil.insert("，") +
            digit_3 + separator_delete + segment_sep +
            digit_3 + separator_delete + segment_sep +
            digit_4
        )
        
        # 合并所有国际电话格式，按匹配优先级排序
        other_international = pynini.union(
            pynutil.add_weight(other_international_flexible, 0.1),   # 最高优先级：灵活格式
            pynutil.add_weight(other_international_extended, 0.2),   # 高优先级：扩展格式
            pynutil.add_weight(other_international_fixed, 0.3)       # 兼容性：固定格式
        )
        
        # 合并国际电话格式
        international_phone = pynini.union(china_international, other_international)
        
        # ============================================
        # 9. 新增：连续电话号码识别（通过连接词连接）
        # ============================================
        
        # 连接词：支持"或"、"和"、"以及"、"及"等
        connector_words = pynini.union(
            "或", "和", "以及", "及", "与", "跟", "还是", "或者"
        )
        
        # 基础电话号码模式（不包含上下文）
        basic_phone_formats = pynini.union(
            continuous_mobile_11,    # 连续11位手机号
            continuous_service_10,   # 连续10位客服号
            continuous_service_5,    # 连续5位银行客服
            international_phone,     # 国际电话
            mobile_format,          # 手机号码
            fixed_3_8,              # 固定电话
            fixed_3_7,
            fixed_4_8,
            fixed_4_7,
            fixed_4_6,
            service_400_800,        # 客服电话
            service_400_800_short,
            emergency_numbers_basic,  # 紧急电话（不带上下文）
            service_numbers_basic,    # 客服电话（不带上下文）
            generic_phone,           # 泛化模式
        )
        
        # 连续电话号码模式：电话号码 + 连接词 + 电话号码
        # 支持多个电话号码的连接：A或B、A和B或C等
        consecutive_phones = (
            basic_phone_formats + 
            pynini.closure(
                # 可选空格 + 连接词 + 可选空格 + 下一个电话号码
                pynini.closure(NEMO_SPACE, 0, 1) + 
                connector_words + 
                pynini.closure(NEMO_SPACE, 0, 1) + 
                basic_phone_formats,
                1, 3  # 最多支持4个电话号码连接（3个连接词）
            )
        )

        # ============================================
        # 10. 组合所有格式（按置信度排序）
        # ============================================
        
        all_phone_formats = pynini.union(
            # 第一层：连续电话号码（最高置信度）- 新增
            pynutil.add_weight(consecutive_phones, 0.05),    # 连续电话号码（最高优先级）
            
            # 第二层：连续数字格式（高置信度）
            pynutil.add_weight(continuous_mobile_11, 0.1),    # 连续11位手机号
            pynutil.add_weight(continuous_service_10, 0.2),   # 连续10位客服号
            pynutil.add_weight(continuous_service_5, 0.3),    # 连续5位银行客服
            
            # 第三层：国际电话（高置信度）
            pynutil.add_weight(international_phone, 0.5),
            
            # 第四层：标准格式电话（高置信度）
            pynutil.add_weight(mobile_format, 0.7),          # 手机号码
            pynutil.add_weight(fixed_3_8, 0.7),              # 固定电话
            pynutil.add_weight(fixed_3_7, 0.7),
            pynutil.add_weight(fixed_4_8, 0.7),
            pynutil.add_weight(fixed_4_7, 0.7),
            pynutil.add_weight(fixed_4_6, 0.7),
            pynutil.add_weight(service_400_800, 0.8),        # 客服电话
            pynutil.add_weight(service_400_800_short, 0.8),
            
            # 第五层：上下文感知的短号码（中等置信度）
            pynutil.add_weight(emergency_with_context, 0.9),    # 紧急电话
            pynutil.add_weight(service_with_context, 0.9),        # 客服电话
            
            # 第六层：泛化模式（低置信度）
            pynutil.add_weight(generic_phone, 1.0),          # 泛化模式
        )
        
        # ============================================
        # 10. 包装为telephone token
        # ============================================
        
        # 可选的国家代码字段
        optional_country_code = pynini.closure(
            pynutil.insert("country_code: \"") + country_codes + pynutil.insert("\" "),
            0, 1
        )
        
        # 主要号码部分
        number_part = pynutil.insert("number_part: \"") + all_phone_formats + pynutil.insert("\"")
        
        # 真正的分机号：只有明确标示为分机的才处理（如：xxx-xxx分机123）
        # 注意：标准固定电话（如010-12345678）不应该有分机号
        explicit_extension = pynini.closure(
            pynutil.delete("分机") + 
            pynutil.insert(" extension: \"") + 
            pynini.closure(digit + digit_sep, 0, 2) + digit +
            pynutil.insert("\""),
            0, 1
        )
        
        # 完整图：移除自动分机号处理，只处理明确标示的分机
        complete_graph = pynini.union(
            # 带国家代码的格式
            optional_country_code + number_part,
            # 纯号码格式
            number_part,
            # 只有明确标示分机的才处理分机
            number_part + explicit_extension
        )
        
        final_graph = self.add_tokens(complete_graph)
        self.fst = final_graph.optimize() 