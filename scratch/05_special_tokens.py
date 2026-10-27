from llm_sdk import Small_LLM_Model

model = Small_LLM_Model()
ids = model.encode("<|im_start>user")[0].tolist()
for id in ids:
    print(f"ID:{id}, Token: {repr(model.decode(id))}")