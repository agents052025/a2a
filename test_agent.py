#!/usr/bin/env python3
"""
Тестовий скрипт для взаємодії з CrewAI агентом через A2A протокол
"""

import requests
import json
import base64
import argparse
from PIL import Image
import io
import os
from datetime import datetime

def test_health(base_url):
    """Перевірка працездатності агента"""
    url = f"{base_url}/a2a/healthz"
    try:
        response = requests.get(url, timeout=10)  # Додано таймаут 10 секунд
        if response.status_code == 200:
            print("✅ Агент працює")
            return True
        else:
            print(f"❌ Помилка: {response.status_code}, {response.text}")
            return False
    except requests.RequestException as e:  # Більш специфічний виняток
        print(f"❌ Помилка підключення: {str(e)}")
        return False

def get_metadata(base_url):
    """Отримання метаданих агента"""
    url = f"{base_url}/a2a/metadata"
    try:
        response = requests.get(url, timeout=10)  # Додано таймаут 10 секунд
        if response.status_code == 200:
            print("✅ Метадані агента:")
            print(json.dumps(response.json(), indent=2))
            return response.json()
        else:
            print(f"❌ Помилка: {response.status_code}, {response.text}")
            return None
    except requests.RequestException as e:  # Більш специфічний виняток
        print(f"❌ Помилка підключення: {str(e)}")
        return None

def generate_image(base_url, prompt, output_dir="./generated_images"):
    """Генерування зображення"""
    url = f"{base_url}/a2a/task"
    
    payload = {
        "input": {
            "prompt": prompt
        }
    }
    
    print(f"📤 Відправляю запит: '{prompt}'")
    
    try:
        response = requests.post(url, json=payload, timeout=30)  # Збільшений таймаут для генерації зображень
        
        if response.status_code != 200:
            print(f"❌ Помилка: {response.status_code}, {response.text}")
            return None
        
        data = response.json()
        print(f"📥 Отримано відповідь: '{data['result']}'")
        
        # Створення директорії, якщо вона не існує
        os.makedirs(output_dir, exist_ok=True)
        
        # Якщо є артефакти (зображення)
        if data["artifacts"]:
            print(f"🖼️ Знайдено {len(data['artifacts'])} артефактів")
            
            # Отримання першого артефакту (зображення)
            artifact = data["artifacts"][0]
            image_data = artifact["data"]
            
            # Декодування Base64 у зображення
            image_bytes = base64.b64decode(image_data)
            
            # Отримання ID зображення з тексту
            image_id = extract_image_id(data["result"])
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if image_id:
                filename = f"{output_dir}/{timestamp}_{image_id}.png"
            else:
                filename = f"{output_dir}/{timestamp}_image.png"
            
            # Збереження зображення на диск
            with open(filename, "wb") as f:
                f.write(image_bytes)
            
            print(f"💾 Зображення збережено як '{filename}'")
            
            # Спроба відкрити зображення (якщо PIL встановлений)
            try:
                image = Image.open(io.BytesIO(image_bytes))
                print(f"📊 Розмір зображення: {image.width}x{image.height}")
            except (IOError, ValueError) as e:  # Більш специфічні винятки для операцій із зображеннями
                print(f"⚠️ Не вдалося відкрити зображення: {str(e)}")
            
            return image_id
        else:
            print("⚠️ У відповіді немає артефактів (зображень)")
            return None
            
    except Exception as e:
        print(f"❌ Помилка підключення: {str(e)}")
        return None

def edit_image(base_url, image_id, edit_prompt, output_dir="./generated_images"):
    """Редагування існуючого зображення"""
    # Формуємо запит з посиланням на ID зображення
    prompt = f"Відредагуй зображення з image_id: {image_id}, {edit_prompt}"
    return generate_image(base_url, prompt, output_dir)

def extract_image_id(text):
    """Витягнення ID зображення з тексту"""
    import re
    
    # Шукаємо патерни типу "ID: abc-123" або "with ID: abc-123"
    match = re.search(r'ID:\s*([a-f0-9-]+)', text, re.IGNORECASE)
    if match:
        return match.group(1)
    
    # Шукаємо UUID в тексті
    uuids = re.findall(r'([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})', text)
    if uuids:
        return uuids[0]
    
    return None

def get_image_by_id(base_url, image_id, output_dir="./generated_images"):
    """Отримання зображення за його ID"""
    url = f"{base_url}/image/{image_id}"
    
    try:
        response = requests.get(url, timeout=10)  # Додано таймаут
        
        if response.status_code != 200:
            print(f"❌ Помилка: {response.status_code}, {response.text}")
            return False
        
        data = response.json()
        
        if "image" in data:
            # Створення директорії, якщо вона не існує
            os.makedirs(output_dir, exist_ok=True)
            
            # Декодування Base64 у зображення
            image_bytes = base64.b64decode(data["image"])
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{output_dir}/{timestamp}_fetched_{image_id}.png"
            
            # Збереження зображення на диск
            with open(filename, "wb") as f:
                f.write(image_bytes)
            
            print(f"💾 Зображення збережено як '{filename}'")
            return True
        else:
            print("⚠️ У відповіді немає зображення")
            return False
            
    except Exception as e:
        print(f"❌ Помилка підключення: {str(e)}")
        return False

def main():
    """Головна функція"""
    parser = argparse.ArgumentParser(description="Тестування CrewAI агента з A2A протоколом")
    parser.add_argument("--url", type=str, default="http://localhost:10001", help="Базовий URL агента")
    parser.add_argument("--output", type=str, default="./generated_images", help="Директорія для збереження зображень")
    
    subparsers = parser.add_subparsers(dest="command", help="Підкоманди")
    
    # Команда health
    subparsers.add_parser("health", help="Перевірка працездатності агента")
    
    # Команда metadata
    subparsers.add_parser("metadata", help="Отримання метаданих агента")
    
    # Команда generate
    generate_parser = subparsers.add_parser("generate", help="Генерування зображення")
    generate_parser.add_argument("prompt", type=str, help="Текстовий запит для генерування зображення")
    
    # Команда edit
    edit_parser = subparsers.add_parser("edit", help="Редагування існуючого зображення")
    edit_parser.add_argument("image_id", type=str, help="ID зображення для редагування")
    edit_parser.add_argument("prompt", type=str, help="Текстовий запит для редагування зображення")
    
    # Команда get
    get_parser = subparsers.add_parser("get", help="Отримання зображення за ID")
    get_parser.add_argument("image_id", type=str, help="ID зображення для отримання")
    
    # Команда demo
    demo_parser = subparsers.add_parser("demo", help="Повна демонстрація можливостей агента")
    demo_parser.add_argument("--prompt", type=str, default="Намалюй красивий пейзаж з горами та озером", 
                          help="Текстовий запит для генерування зображення")
    demo_parser.add_argument("--edit-prompt", type=str, default="додай на небі веселку", 
                          help="Текстовий запит для редагування зображення")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == "health":
        test_health(args.url)
    
    elif args.command == "metadata":
        get_metadata(args.url)
    
    elif args.command == "generate":
        generate_image(args.url, args.prompt, args.output)
    
    elif args.command == "edit":
        edit_image(args.url, args.image_id, args.prompt, args.output)
    
    elif args.command == "get":
        get_image_by_id(args.url, args.image_id, args.output)
    
    elif args.command == "demo":
        print("🚀 Запуск демонстрації можливостей агента")
        print("-" * 50)
        
        # Перевірка працездатності
        if not test_health(args.url):
            return
        
        print("-" * 50)
        
        # Отримання метаданих
        get_metadata(args.url)
        
        print("-" * 50)
        
        # Генерування зображення
        print("🎨 Генерування нового зображення")
        image_id = generate_image(args.url, args.prompt, args.output)
        
        if not image_id:
            print("❌ Не вдалося згенерувати зображення")
            return
        
        print("-" * 50)
        
        # Редагування зображення
        print(f"✏️ Редагування зображення з ID: {image_id}")
        edited_image_id = edit_image(args.url, image_id, args.edit_prompt, args.output)
        
        print("-" * 50)
        
        # Отримання зображення за ID
        if edited_image_id:
            print(f"🔍 Отримання зображення за ID: {edited_image_id}")
            get_image_by_id(args.url, edited_image_id, args.output)
        
        print("-" * 50)
        print("✨ Демонстрація завершена")

if __name__ == "__main__":
    main()
