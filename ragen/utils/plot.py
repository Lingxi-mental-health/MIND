import base64
from io import BytesIO
from typing import List
from PIL import Image
import os
import re
import html

def image_to_base64(img_array):
    """Convert numpy array to base64 string"""
    img = Image.fromarray(img_array)
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def save_trajectory_to_output(trajectory, save_dir) -> List[str]:
    """
    Save the trajectory to HTML files with better multi-language support
    
    Arguments:
        - trajectory (list): The trajectory to save
        - save_dir (str): Directory to save the HTML files
    
    Returns:
        - filenames (list): List of saved HTML file paths
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # CSS styles as a separate string
    css_styles = '''
        body {
            font-family: "Noto Sans CJK SC", "Noto Sans CJK JP", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .trajectory-step {
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .section {
            margin-bottom: 12px;
        }
        .section-title {
            font-size: 15px;
            font-weight: 600;
            color: #555;
            margin-bottom: 6px;
        }
        .section-content {
            background: #fafafa;
            border-radius: 4px;
            border: 1px solid #e0e0e0;
            padding: 10px 12px;
            white-space: pre-wrap;
            font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
            font-size: 13px;
            line-height: 1.5;
            color: #222;
        }
        .section-content.think {
            background: #fff8e1;
            border-color: #ffcc80;
        }
        .section-content.answer {
            background: #e8f5e9;
            border-color: #81c784;
        }
        .section-content.full-output {
            background: #e3f2fd;
            border-color: #64b5f6;
        }
        .section-content.error {
            background: #ffebee;
            border-color: #ef9a9a;
        }
        .image-container {
            display: flex;
            justify-content: space-between;
            margin-bottom: 15px;
        }
        .image-box {
            width: 48%;
        }
        .image-box img {
            width: 100%;
            height: auto;
            border-radius: 4px;
        }
        .image-title {
            font-size: 16px;
            font-weight: bold;
            margin-bottom: 10px;
            color: #333;
        }
        .response-box {
            background: #f8f8f8;
            border-radius: 4px;
            padding: 15px;
            margin: 10px 0;
            white-space: pre-wrap;
            font-size: 14px;
            line-height: 1.5;
        }
        .step-number {
            font-size: 18px;
            font-weight: bold;
            color: #2196F3;
            margin-bottom: 15px;
        }
        .tag {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 600;
            margin-left: 8px;
        }
        .tag-think { background: #fff8e1; color: #f57c00; }
        .tag-answer { background: #e8f5e9; color: #388e3c; }
        .tag-error { background: #ffebee; color: #d32f2f; }
    '''
    
    filenames = []
    for data_idx, data in enumerate(trajectory):
        steps_html = []
        step_entries = data.get('steps')
        
        def build_section(title, content, css_class=""):
            if not content:
                return ""
            class_attr = f'section-content {css_class}' if css_class else 'section-content'
            return f'''
            <div class="section">
                <div class="section-title">{title}</div>
                <div class="{class_attr}">{content}</div>
            </div>
            '''
        
        def extract_answer_text(text):
            match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
            return match.group(1).strip() if match else ""
        
        def check_format_error(patient_feedback):
            """检查是否有格式错误"""
            if not patient_feedback:
                return False
            return "格式错误" in patient_feedback or "标签错误" in patient_feedback
        
        if step_entries:
            for step in step_entries:
                prompt_html = html.escape(step.get("prompt", "") or "")
                observation_html = html.escape(step.get("observation", "") or "")
                thought_text = step.get("thought", "")
                answer_text = step.get("answer", "") or extract_answer_text(step.get("response", "") or "")
                response_raw = step.get("response", "")
                patient_feedback = step.get("patient_status", "") or ""
                
                thought_html = html.escape(thought_text) if thought_text else ""
                answer_html = html.escape(answer_text) if answer_text else ""
                
                # 格式化完整输出，保留换行和标签结构
                response_clean = response_raw.replace("<|im_end|>", "").replace("<|endoftext|>", "")
                response_html = html.escape(response_clean)
                
                # 检查是否有格式错误
                has_error = check_format_error(patient_feedback)
                patient_feedback_html = html.escape(patient_feedback)
                feedback_class = "error" if has_error else ""
                
                # 构建状态标签
                status_tags = ""
                if thought_text:
                    status_tags += '<span class="tag tag-think">有思考</span>'
                if answer_text:
                    status_tags += '<span class="tag tag-answer">有回答</span>'
                if has_error:
                    status_tags += '<span class="tag tag-error">格式错误</span>'
                
                sections = [
                    build_section("患者反馈", patient_feedback_html, feedback_class),
                    build_section("医生思考 (Think)", thought_html, "think") if thought_html else "",
                    build_section("医生回答 (Answer)", answer_html, "answer") if answer_html else "",
                    build_section("医生完整生成", response_html, "full-output"),
                    build_section("输入 Prompt", prompt_html),
                ]
                
                step_html = f'''
                <div class="trajectory-step">
                    <div class="step-number">Step {step.get("turn", 0)} {status_tags}</div>
                    {''.join(sections)}
                </div>
                '''
                steps_html.append(step_html)
        else:
            # 兼容旧结构
            n_steps = len(data.get('state', []))
            for step in range(n_steps):
                image_state = data['state'][step]
                parsed_response = data['parsed_response'][step]['raw']
                parsed_response = parsed_response.replace("<|im_end|>", "").replace("<|endoftext|>", "")
                parsed_response = parsed_response.replace('\\n', '\n')
                parsed_response = re.sub(r'(</think>)\s*(<answer>)', r'\1\n\2', parsed_response)
                parsed_response = html.escape(parsed_response)
                
                state_html = f'<div class="section-content">{html.escape(image_state)}</div>'
                
                step_html = f'''
                <div class="trajectory-step">
                    <div class="step-number">Step {step + 1}</div>
                    <div class="section">
                        <div class="section-title">Observation</div>
                        {state_html}
                    </div>
                    <div class="section">
                        <div class="section-title">Model Output</div>
                        <div class="section-content">{parsed_response}</div>
                    </div>
                </div>
                '''
                steps_html.append(step_html)
        
        # Combine all content into final HTML
        final_html = f'''
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                {css_styles}
            </style>
        </head>
        <body>
            {''.join(steps_html)}
        </body>
        </html>
        '''
        
        # Save to file
        filename = os.path.join(save_dir, f"trajectory_data_{data_idx}.html")
        filenames.append(filename)
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(final_html)
    return filenames
        
        

def parse_llm_output(llm_output: str, strategy: str):
    """
    Parse the llm output
    =========================
    Arguments:
        - llm_output: the output of the llm
        - strategy: the strategy to parse the llm output
            - "formated": parse action and thought
            - "raw": parse the raw output
    =========================
    Returns (dict): parsed output
    """
    if strategy == "raw":
        return {'raw': llm_output}
    elif strategy == "formatted":
        if "<answer>" in llm_output:
            action = llm_output.split("<answer>")[1].split("</answer>")[0].strip()
        else:
            action = ""
        if "<think>" in llm_output:
            thought = llm_output.split("<think>")[1].split("</think>")[0].strip()
        else:
            thought = ""
        return {
            'action': action,
            'thought': thought
        }
    else:
        raise ValueError(f"Unknown strategy: {strategy}")