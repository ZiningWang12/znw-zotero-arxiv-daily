import os
from openai import OpenAI
api_key = "AIzaSyDijBhdh2Hj-D6C-iaTatB_5zvX9C5YRi0"
''' OPENAI API 调用
'''
try:
    # 使用你的 Gemini API 密钥 (不要加 sk-)
    # 确保 OPENAI_API_KEY 环境变量设置的是你的 Gemini API 密钥
    if not api_key:
        # 如果环境变量中没有，请直接在这里填入你的 Gemini API Key
        api_key = "YOUR_ACTUAL_GEMINI_API_KEY" # <--- 把这里替换成你的真实密钥
        if api_key == "YOUR_ACTUAL_GEMINI_API_KEY":
            print("错误：请替换脚本中的 'YOUR_ACTUAL_GEMINI_API_KEY' 为你的真实 Gemini API 密钥，或者设置 OPENAI_API_KEY 环境变量。")
            exit()

    client = OpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )

    # 从 Google SDK 的输出中，我们知道 'models/gemini-2.0-flash' 是有效的
    # 对于 OpenAI 库，我们通常只用 'gemini-2.0-flash'
    model_name_to_test = "gemini-2.0-flash"
    # 你也可以尝试 "gemini-1.5-flash"
    # model_name_to_test = "gemini-1.5-flash"

    print(f"正在尝试使用 OpenAI 库连接到模型: {model_name_to_test}，通过 URL: {client.base_url}")

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "你好！请用一句话介绍一下你自己。",
            }
        ],
        model=model_name_to_test,
        temperature=0.7,
    )

    print("\nOpenAI 库 API 调用成功！")
    print("来自模型的回应:")
    print(chat_completion.choices[0].message.content)

except Exception as e:
    print(f"\nOpenAI 库 API 调用失败。错误信息如下:")
    print(e)
    # 可以在这里添加更详细的错误类型判断
    if "Connection error" in str(e):
        print("提示：仍然是连接错误。请确保 Python 环境的网络设置（如代理）与 curl 时一致。")
    elif "APIStatusError" in str(e) or "APIError" in str(e) or "BadRequestError" in str(e):
        print("提示：收到了 API 错误。这可能与请求的格式、模型名称或 base_url 有关。请检查错误详情。")
        print(f"详细错误类型: {type(e)}")

''' GOOGLE SDK API 调用
import google.generativeai as genai
import os

try:
    # 确保你在这里使用的是你真实的 Gemini API Key，不要加 sk-
    # 最好设置为环境变量 GEMINI_API_KEY，或者直接在这里赋值
    if not api_key:
        # 如果环境变量中没有，请直接在这里填入你的 API Key
        api_key = "YOUR_ACTUAL_GEMINI_API_KEY" # <--- 把这里替换成你的真实密钥
        if api_key == "YOUR_ACTUAL_GEMINI_API_KEY":
            print("错误：请替换脚本中的 'YOUR_ACTUAL_GEMINI_API_KEY' 为你的真实 Gemini API 密钥，或者设置 GEMINI_API_KEY 环境变量。")
            exit()

    genai.configure(api_key=api_key)

    print("配置 Google SDK 完成。")
    print("正在尝试使用 Google SDK 列出可用模型...")
    model_found_for_content_generation = False
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"找到支持 generateContent 的模型: {m.name}")
            model_found_for_content_generation = True

    if not model_found_for_content_generation:
        print("警告：未能通过 list_models() 找到任何明确支持 'generateContent' 的模型。可能是 API 密钥权限问题或没有适合的模型。")
        print("仍然尝试使用 'gemini-1.5-flash' 进行内容生成测试...")
    else:
        print("\n模型列出成功。")

    print("\n现在尝试使用 Google SDK 生成内容 (gemini-1.5-flash)...")
    model = genai.GenerativeModel('gemini-1.5-flash') # 确保你的 API key 有权访问此模型

    response = model.generate_content("你好，请用一句话介绍你自己。")

    print("\nGoogle SDK API 调用成功！")
    print("来自模型的回应:")
    print(response.text)

except Exception as e:
    print(f"\nGoogle SDK API 调用失败。错误信息如下:")
    print(e)
    error_str = str(e).lower()
    if "api_key_invalid" in error_str or ("permission" in error_str and "denied" in error_str):
        print("\n提示：API 密钥无效或权限不足。请仔细检查你的 Gemini API 密钥，确保它有效、已启用，并且你的 Google Cloud 项目已正确设置（包括必要的 API 启用和结算）。")
    elif "defaultcredentialserror" in error_str:
        print("\n提示：未能找到默认凭据。请确保 API 密钥已正确配置。")
    elif "deadline_exceeded" in error_str or "unavailable" in error_str or "connection refused" in error_str or "dns failure" in error_str:
        print("\n提示：连接超时或服务不可用。这可能表示网络问题、Google 服务临时中断或防火墙/代理问题。请检查你的网络连接。")
    elif "not found" in error_str and "model" in error_str:
        print("\n提示：模型未找到。请确保 'gemini-1.5-flash' 或你选择的模型对你的 API 密钥可用。")

'''