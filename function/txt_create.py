# 生成300行的文本
with open('output.txt', 'w') as file:
    for i in range(1, 31557):
        filename = f"img{str(i).zfill(6)}.png"
        # file.write(f"{filename} {filename}\n")
        file.write(f"{filename} {filename} {filename}\n")
