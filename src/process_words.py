import os
import re
import time
import json
import pandas as pd
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

# 添加环境变量处理
import sys
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# 从环境变量获取API密钥
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    logger.error("DeepSeek API密钥未设置! 请通过环境变量DEEPSEEK_API_KEY提供")
    sys.exit(1)

# 其他配置
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "20"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))

# 初始化DeepSeek客户端
client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com"
)

# 进度文件路径
PROGRESS_FILE = "progress.json"
RESULTS_DIR = "results"
ERRORS_DIR = "errors"

# 确保目录存在
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(ERRORS_DIR, exist_ok=True)

def load_progress():
    """加载处理进度"""
    try:
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"last_index": 0, "processed": 0, "batches": []}

def save_progress(progress):
    """保存处理进度"""
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)

def load_wordlist():
    """加载词表文件"""
    # 尝试不同可能的路径
    possible_paths = [
        "src/wordlist/sample_new_words.csv",
        "wordlist/sample_new_words.csv",
        "sample_new_words.csv"
    ]
    
    for path in possible_paths:
        try:
            df = pd.read_csv(path, encoding="utf-8-sig")
            print(f"✅ 成功加载词表: {path}")
            return df["word"].tolist()
        except FileNotFoundError:
            continue
    
    raise FileNotFoundError("未找到词表文件。请确保文件位于 src/wordlist/ 或项目根目录")

def build_prompt(word_batch):
    """构建提示词模板"""
    prompt = """你是一位中文语言演化与心理语言学专家，负责评估给定词语是否为2008年后出现的新词。特别注意避免认知偏差，通过该词在2008年前的含义客观对比变化，优先基于语义和语用功能的实质性证据（如语料库或演化研究），而非高频印象。在判断时，积极考虑反例（如历史用例或未被接受的新义），挑战初步假设，确保结论基于客观证据而非预设偏好。严格按顺序完成以下步骤，禁止跳步：

步骤1：语义变化判断
   - 反事实思考：对比2008年前后的核心含义，是否有新的核心含义或语用功能（必须是含义的本质变化，而非仅场景/载体变化的领域延伸，如“刷”从“刷牙”到“刷手机”是场景变化，但核心“快速操作”不变；而“躺平”新增“消极抵抗”含义属本质变化）。
   - 如是，进入“A 旧词新义”候选；如否，进入下一步。

步骤2：形式判断
   - 判断该词形式是否为新造（包括构词创新、中英文缩略、谐音、拼音、舶来等），即组合方式不符合传统构词法，或者含义无法从字面推导。
   - 如是，进入“B 新词新义”候选；如否，进入下一步。

步骤3：排除法
   - 若满足以下任一条件，考虑判为“C 非新词”：
     1) 形式与含义均沿用传统（如“书”无论纸质或电子，核心“阅读载体”不变）。
     2) 核心含义未变，仅场景迁移/载体变化/搭配拓展（如“窗口”从物体到电子的聊天窗口，核心含义不变）。
     3) 新出现的专有名词，且未衍生出普通词汇用法（品牌、人名、地名、作品名等，如“微信”虽新但属品牌名）。
     4) 由政府/广告推出但未被民间广泛使用，未发生语义/形式创新（如官方术语“寡头”语义不变）。
     5) 无可靠证据支持语义或形式新变（基于步骤1-2分析）。

步骤4：边界判断
   - 如果分析后仍不确定（如证据冲突、使用范围有限），或该词位于新旧交界（如旧词新义但变化轻微），标记为边界词。

输出格式（严格保持顺序和行数，不添加多余解释）：
原始词语

分类简要理由（先陈述反事实思考的语义对比和形式分析，再说明排除或归类原因）

两个语义邻近词（用逗号分隔）

分类结果（A / B / C）

对“是否为A或B类新词”的判断置信度（1-10，基于证据强度：10=高证据，如多个语料来源；1=低证据，如依赖推测）

是否为边界词（是 / 否）

输出示例
摸鱼
原表示“捕鱼”，当前新增“工作中偷懒”的含义，属语义本质变化，形式为旧。
偷懒,划水
A
9
否

- 每组词之间用换行分隔。
- 不要添加任何说明文字或编号。保持输出结构统一。
"""

    for i, word in enumerate(word_batch, 1):
        prompt += f"\n词语 {i}：{word}"
    return prompt

@retry(stop=stop_after_attempt(MAX_RETRIES), wait=wait_random_exponential(multiplier=1, max=60))
def process_batch(batch, batch_num):
    """处理单个批次"""
    prompt = build_prompt(batch)
    
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一位中文语言学家"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=10000
        )
        return response.choices[0].message.content
    except Exception as e:
        raise Exception(f"DeepSeek API调用失败: {str(e)}")

def parse_response(response_text, batch):
    """解析API响应并处理错误"""
    entries = []
    raw_blocks = response_text.strip().split('\n\n')
    
    # 有效响应模式
    valid_pattern = re.compile(r".+\n.+\n.+\n[A-C]\n\d+\n(是|否)")
    
    for idx, block in enumerate(raw_blocks):
        try:
            # 检查是否为有效响应格式
            block = block.strip()
            if not valid_pattern.match(block):
                raise ValueError("响应格式不符合预期")
                
            lines = block.split('\n')
            if len(lines) < 6:
                raise ValueError("响应行数不足")
                
            entry = {
                "word": batch[idx] if idx < len(batch) else f"UNKNOWN_{idx}",
                "reason": lines[1],
                "near_words": lines[2],
                "category": lines[3],
                "confidence": lines[4],
                "is_boundary": lines[5],
                "raw_response": block
            }
            entries.append(entry)
        except Exception as e:
            # 标记错误条目
            word = batch[idx] if idx < len(batch) else f"UNKNOWN_{idx}"
            entries.append({
                "word": word,
                "error": f"解析错误: {str(e)}",
                "raw_response": block
            })
    
    # 检查是否有遗漏的词语
    processed_words = {entry["word"] for entry in entries}
    for word in batch:
        if word not in processed_words:
            entries.append({
                "word": word,
                "error": "未在响应中找到对应结果",
                "raw_response": response_text[:200] + "..."  # 截取部分响应
            })
    
    return entries

def save_results(entries, batch_num):
    """保存结果到CSV文件"""
    filename = f"{RESULTS_DIR}/batch_{batch_num}.csv"
    df = pd.DataFrame(entries)
    df.to_csv(filename, index=False, encoding="utf-8-sig")
    return filename

def save_errors(entries, batch_num):
    """保存错误信息"""
    filename = f"{ERRORS_DIR}/batch_{batch_num}_errors.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    return filename

def aggregate_results():
    """聚合所有批次结果"""
    all_entries = []
    
    # 遍历结果目录
    for filename in os.listdir(RESULTS_DIR):
        if filename.endswith(".csv"):
            filepath = os.path.join(RESULTS_DIR, filename)
            df = pd.read_csv(filepath)
            all_entries.extend(df.to_dict("records"))
    
    # 创建最终结果文件
    final_filename = "final_results.csv"
    pd.DataFrame(all_entries).to_csv(final_filename, index=False, encoding="utf-8-sig")
    
    # 统计信息
    total_words = len(all_entries)
    success_words = sum(1 for e in all_entries if "error" not in e or pd.isna(e.get("error")))
    error_words = total_words - success_words
    
    return {
        "filename": final_filename,
        "total_words": total_words,
        "success_words": success_words,
        "error_words": error_words
    }

def main():
    """主处理函数"""
    print("🚀 开始处理词表...")
    start_time = time.time()
    
    # 加载词表
    words = load_wordlist()
    total_words = len(words)
    print(f"📋 词表加载完成，共 {total_words} 个词语")
    
    # 加载进度
    progress = load_progress()
    start_index = progress["last_index"]
    processed_batches = set(progress["batches"])
    
    print(f"⏱️ 从第 {start_index} 个词语继续处理...")
    
    # 分批处理
    for i in range(start_index, total_words, BATCH_SIZE):
        batch_num = (i // BATCH_SIZE) + 1
        
        # 跳过已处理的批次
        if batch_num in processed_batches:
            print(f"⏭️ 批次 #{batch_num} 已处理，跳过")
            continue
            
        batch = words[i:i+BATCH_SIZE]
        print(f"\n🔍 处理批次 #{batch_num}: {batch}")
        
        try:
            # 处理批次
            response_text = process_batch(batch, batch_num)
            
            # 解析结果
            entries = parse_response(response_text, batch)
            
            # 统计成功/失败
            success_count = sum(1 for e in entries if "error" not in e)
            error_count = len(entries) - success_count
            
            # 保存结果
            result_file = save_results(entries, batch_num)
            print(f"✅ 批次 #{batch_num} 完成! 成功: {success_count}, 失败: {error_count}")
            print(f"💾 结果保存至: {result_file}")
            
            # 如果有错误，单独保存错误信息
            if error_count > 0:
                error_entries = [e for e in entries if "error" in e]
                error_file = save_errors(error_entries, batch_num)
                print(f"⚠️ 发现 {error_count} 个错误，保存至: {error_file}")
            
            # 更新进度
            progress["last_index"] = i + BATCH_SIZE
            progress["processed"] += len(batch)
            progress["batches"].append(batch_num)
            save_progress(progress)
            
        except Exception as e:
            print(f"❌ 批次 #{batch_num} 失败: {str(e)}")
            # 保存错误信息
            error_entries = [{"word": w, "error": str(e)} for w in batch]
            error_file = save_errors(error_entries, batch_num)
            print(f"💾 错误信息保存至: {error_file}")
            
            # 更新进度（跳过当前批次）
            progress["last_index"] = i + BATCH_SIZE
            progress["batches"].append(batch_num)
            save_progress(progress)
        
        # 避免速率限制
        time.sleep(1)
    
    # 聚合结果
    if progress["last_index"] >= total_words:
        print("\n✨ 所有批次处理完成! 正在聚合结果...")
        agg_result = aggregate_results()
        print(f"📊 处理统计: 总词语 {agg_result['total_words']}, "
              f"成功 {agg_result['success_words']}, "
              f"失败 {agg_result['error_words']}")
        print(f"💾 最终结果保存至: {agg_result['filename']}")
    
    # 计算总耗时
    elapsed = time.time() - start_time
    print(f"\n⏱️ 总耗时: {elapsed:.2f}秒")
    print("✅ 处理完成!")

if __name__ == "__main__":
    main()
