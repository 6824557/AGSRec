def merge_contacts(num, records):
    contact_dict = {}  # 用于存储联系人姓名和手机号列表

    for record in records:
        # 分割记录，获取姓名和手机号
        data = record.split()
        name = data[0]
        phones = set(data[1:])  # 使用集合去除重复的手机号

        # 检查手机号是否已存在，若存在则合并
        found = False
        for phone in phones:
            for existing_name, existing_phones in contact_dict.items():
                if phone in existing_phones:
                    # 合并两个联系人的手机号
                    contact_dict[existing_name].update(phones)
                    found = True
                    break
            if found:
                break
        if not found:
            if name in contact_dict:
                # 如果同名联系人已存在，合并手机号
                contact_dict[name].update(phones)
            else:
                contact_dict[name] = phones

    # 将联系人及其手机号排序
    sorted_contacts = sorted((name, sorted(list(phones))) for name, phones in contact_dict.items())

    # 输出整理后的通讯录
    for name, phones in sorted_contacts:
        print(f"{name} {' '.join(phones)}")

# 输入处理
num = int(input())  # 获取记录数量
records = [input().strip() for _ in range(num)]  # 获取通讯录记录

merge_contacts(num, records)
