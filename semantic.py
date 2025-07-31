import torch
from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel, BertModel

# 加载GPT-2模型和分词器
# tokenizer = AutoTokenizer.from_pretrained('./template_model/gpt2')
# model = AutoModelForCausalLM.from_pretrained('./template_model/gpt2')
# 加载预训练的GPT-2模型和分词器
model_name = '/usr/workspace/dxn/SAFL/SAFL/template_model/gpt2'  # 可根据实际需求选择不同的GPT-2模型
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
sem_model = GPT2Model.from_pretrained(model_name)
sem_model.eval()
for param in sem_model.parameters():
    param.requires_grad = False


def str2bool(s):
  if s not in {'false', 'true'}:
    raise ValueError('Not a valid boolean string')
  return s == 'true'


def convert_timeid2text(day, week):
    day_map = ['**','midnight', 'early morning', 'noon', 'afternoon', 'night']
    week_map = ['**','monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
    return day_map[day], week_map[week]


def get_traj_prompt(traj_data,loc_sem,semlength):
    u_traj = traj_data  # 用户u的轨迹数据
    max_len = semlength
    template = ""
    if len(u_traj) < max_len:
        max_len = len(u_traj)
    for i in range(max_len, 0, -1):
        loc = u_traj[-i][1]  # 用户u最近访问的地点i
        loc_info = loc_sem[loc][0]  # 地点i的语义信息
        # 提取相关信息
        weekday = u_traj[-i][4]  # 访问的星期几
        time_category = u_traj[-i][5]  # 访问的时间类别
        c_name = u_traj[-i][3]  # 地点i的类型
        top_3_users = loc_info[0]  # 最常来此地的用户
        top_3_cates = loc_info[1]  # 这些用户最常去的地点类型
        max_time = loc_info[2]  # 人们最常访问此地的时间
        max_zoneCate = loc_info[3]  # 该地点所在区域最多的地点类型
        prompt = (
            f"User {u_traj[0][0]} visited location {loc} on a {weekday} during {time_category}. "
            f"This location is of type {c_name}. "
            f"The top 3 users who frequently visit this location are {top_3_users[0][0]}, {top_3_users[1][0]}, and {top_3_users[2][0]}. "
            f"Their most frequently visited location types are {top_3_cates[0][0]}, {top_3_cates[1][0]}, and {top_3_cates[2][0]}. "
            f"People most often visit this location during {max_time}. "
            f"The most common location type in this area is {max_zoneCate}."
        )

        template = template + prompt
    return template


# 定义一个函数来为轨迹生成语义向�?
# GPT2
def generate_traj_vector(traj_prompt, device):
    # 将轨迹信息格式化为prompt字符�?
    prompt = traj_prompt
    cls = torch.tensor([50256])
    sem = tokenizer.encode(prompt)
    input_ids = torch.tensor(tokenizer.encode(prompt)) # [bs, 20*seq_len]
    input_ids = torch.cat((input_ids, cls), dim=-1)
    with torch.no_grad():
        outputs = sem_model(input_ids)#GPT2 [bs , 20*seq_len, 768]
    semantic_vector = outputs.last_hidden_state[-1, :]  # bs, 1, 768
    return semantic_vector, sem
