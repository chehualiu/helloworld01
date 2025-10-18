#!/usr/bin/env python3
"""
从Windows的Chrome浏览器获取特定URL的cookie
"""

import base64
import ctypes
import ctypes.wintypes
import contextlib
import hashlib
import http.cookiejar
import io
import json
import os
import re
import shutil
import sqlite3
import struct
import sys
import tempfile


def get_chrome_cookies_for_url(url):
    """
    从Windows的Chrome浏览器获取特定URL的cookie

    Args:
        url: 要获取cookie的URL

    Returns:
        dict: 包含cookie名称和值的字典
    """
    # 确保只在Windows上运行
    if sys.platform not in ('win32', 'cygwin'):
        print("错误: 此脚本仅在Windows系统上运行")
        return {}

    try:
        # 获取Chrome用户数据目录
        chrome_user_data_dir = os.path.join(os.path.expandvars('%LOCALAPPDATA%'), r'Google\Chrome\User Data')

        # 查找Cookies数据库
        cookies_db_path = find_newest_cookies_db(chrome_user_data_dir)
        if not cookies_db_path:
            print("错误: 找不到Chrome Cookies数据库")
            return {}

        # 获取解密密钥
        key = get_windows_chrome_key(chrome_user_data_dir)
        if not key:
            print("错误: 无法获取解密密钥")
            return {}

        # 解析URL获取域名
        domain = extract_domain(url)

        # 从数据库中提取cookie
        cookies = extract_cookies_from_db(cookies_db_path, domain, key)

        return cookies
    except Exception as e:
        print(f"错误: {str(e)}")
        return {}


def find_newest_cookies_db(user_data_dir):
    """
    在Chrome用户数据目录中查找最新的Cookies数据库文件

    Args:
        user_data_dir: Chrome用户数据目录路径

    Returns:
        str: Cookies数据库文件路径，如果未找到则返回None
    """
    newest_db = None
    newest_mtime = 0

    # 遍历用户数据目录下的所有文件夹查找Cookies文件
    for root, _, files in os.walk(user_data_dir):
        if 'Cookies' in files:
            db_path = os.path.join(root, 'Cookies')
            mtime = os.path.getmtime(db_path)
            if mtime > newest_mtime:
                newest_mtime = mtime
                newest_db = db_path

    return newest_db


def get_windows_chrome_key(user_data_dir):
    """
    从Chrome的Local State文件中获取并解密主密钥

    Args:
        user_data_dir: Chrome用户数据目录路径

    Returns:
        bytes: 解密后的密钥，如果获取失败则返回None
    """
    # 查找Local State文件
    local_state_path = os.path.join(user_data_dir, 'Local State')
    if not os.path.exists(local_state_path):
        # 尝试在默认配置文件目录下查找
        local_state_path = os.path.join(user_data_dir, 'Default', 'Local State')
        if not os.path.exists(local_state_path):
            return None

    # 读取Local State文件
    try:
        with open(local_state_path, 'r', encoding='utf-8') as f:
            local_state = json.load(f)

        # 提取加密的密钥
        encrypted_key = base64.b64decode(local_state['os_crypt']['encrypted_key'])

        # 移除DPAPI前缀
        if encrypted_key.startswith(b'DPAPI'):
            encrypted_key = encrypted_key[5:]
            # 使用DPAPI解密
            return decrypt_windows_dpapi(encrypted_key)

        return None
    except Exception:
        return None


def decrypt_windows_dpapi(encrypted_data):
    """
    使用Windows DPAPI解密数据

    Args:
        encrypted_data: 要解密的数据

    Returns:
        bytes: 解密后的数据
    """

    class DATA_BLOB(ctypes.Structure):
        _fields_ = [('cbData', ctypes.wintypes.DWORD),
                    ('pbData', ctypes.POINTER(ctypes.c_char))]

    buffer = ctypes.create_string_buffer(encrypted_data)
    blob_in = DATA_BLOB(len(encrypted_data), buffer)
    blob_out = DATA_BLOB()

    # 调用Windows API解密数据
    result = ctypes.windll.crypt32.CryptUnprotectData(
        ctypes.byref(blob_in),
        None,
        None,
        None,
        None,
        0,
        ctypes.byref(blob_out)
    )

    if not result:
        raise Exception("DPAPI解密失败")

    # 提取解密后的数据
    decrypted_data = ctypes.string_at(blob_out.pbData, blob_out.cbData)

    # 释放内存
    ctypes.windll.kernel32.LocalFree(blob_out.pbData)

    return decrypted_data


def extract_domain(url):
    """
    从URL中提取域名

    Args:
        url: 输入的URL

    Returns:
        str: 提取的域名
    """
    # 移除协议和路径
    domain = re.sub(r'^https?://', '', url)
    domain = re.sub(r'/.*$', '', domain)

    # 提取主域名
    parts = domain.split('.')
    if len(parts) >= 2:
        # 对于.com.cn等多级域名的特殊处理
        if len(parts) >= 3 and parts[-2] in ('com', 'net', 'org', 'gov', 'edu') and len(parts[-1]) == 2:
            return '.'.join(parts[-3:])
        return '.'.join(parts[-2:])

    return domain


def extract_cookies_from_db(db_path, domain, key):
    cookies = {}

    # 创建临时目录复制数据库文件，避免锁定
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_db_path = os.path.join(tmpdir, 'Cookies')

        # 尝试复制文件，添加重试逻辑和自动关闭Chrome功能
        file_copied = False

        # 第一次尝试：正常重试
        max_attempts = 2
        attempt = 0
        while attempt < max_attempts:
            try:
                # 使用二进制模式打开文件进行复制，避免锁定问题
                with open(db_path, 'rb') as src_file, open(temp_db_path, 'wb') as dst_file:
                    # 分块复制文件
                    buffer_size = 1024 * 1024  # 1MB缓冲区
                    while True:
                        buffer = src_file.read(buffer_size)
                        if not buffer:
                            break
                        dst_file.write(buffer)
                file_copied = True
                break  # 成功复制后退出循环
            except (PermissionError, IOError):
                attempt += 1
                if attempt < max_attempts:
                    # 短暂等待后重试
                    import time
                    time.sleep(0.5)

        # 如果常规重试失败，提供自动关闭Chrome选项
        if not file_copied:
            try:
                import subprocess

                # 检查是否安装了psutil库
                has_psutil = False
                try:
                    import psutil
                    has_psutil = True
                except ImportError:
                    pass

                # 询问用户是否要关闭Chrome
                # response = input(
                #     "无法访问Cookies文件，Chrome浏览器可能正在运行。是否自动关闭Chrome浏览器？(y/n): ").strip().lower()
                if True:
                    print("正在关闭Chrome浏览器...")

                    if has_psutil:
                        # 使用psutil关闭Chrome进程
                        chrome_processes = [proc for proc in psutil.process_iter(['name'])
                                            if proc.info['name'] == 'chrome.exe']
                        for proc in chrome_processes:
                            proc.terminate()
                        # 等待进程终止
                        if chrome_processes:
                            psutil.wait_procs(chrome_processes, timeout=5)
                    else:
                        # 使用Windows命令行关闭Chrome
                        try:
                            subprocess.run(['taskkill', '/f', '/im', 'chrome.exe'],
                                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                        except Exception:
                            print("警告: 无法使用命令行关闭Chrome")

                    print("Chrome浏览器已关闭，请稍候...")
                    import time
                    time.sleep(2)  # 给系统一点时间释放文件锁

                    # 再次尝试复制文件
                    try:
                        with open(db_path, 'rb') as src_file, open(temp_db_path, 'wb') as dst_file:
                            buffer_size = 1024 * 1024  # 1MB缓冲区
                            while True:
                                buffer = src_file.read(buffer_size)
                                if not buffer:
                                    break
                                dst_file.write(buffer)
                        file_copied = True
                    except Exception as retry_error:
                        print(f"关闭Chrome后重试失败: {str(retry_error)}")
                        print("提示: 请手动关闭Chrome浏览器后再试。")
                        return {}
                else:
                    print("提示: 请手动关闭Chrome浏览器后再试。")
                    return {}
            except Exception as inner_error:
                print(f"尝试自动关闭Chrome时出错: {str(inner_error)}")
                print("提示: 请手动关闭Chrome浏览器后再试。")
                return {}

        # 如果文件复制成功，继续处理数据库
        if file_copied:
            # 连接数据库
            try:
                conn = sqlite3.connect(temp_db_path)
                cursor = conn.cursor()

                try:
                    # 获取数据库版本
                    cursor.execute('SELECT value FROM meta WHERE key = "version"')
                    row = cursor.fetchone()
                    meta_version = int(row[0]) if row else 0

                    # 查询与指定域名相关的cookie
                    cursor.execute(
                        'SELECT name, value, encrypted_value FROM cookies WHERE host_key LIKE ? OR host_key LIKE ?',
                        (f'%{domain}', f'.{domain}')
                    )

                    # 处理查询结果
                    for name, value, encrypted_value in cursor.fetchall():
                        # 如果有加密值则使用解密后的值
                        if encrypted_value and len(encrypted_value) > 0:
                            decrypted_value = decrypt_chrome_cookie(encrypted_value, key, meta_version)
                            if decrypted_value:
                                cookies[name] = decrypted_value
                        elif value:
                            cookies[name] = value
                finally:
                    # 关闭数据库连接
                    conn.close()
            except sqlite3.Error as e:
                print(f"错误: 无法读取Cookies数据库: {str(e)}")
                return {}

    return cookies


def decrypt_chrome_cookie(encrypted_value, key, meta_version):
    """
    解密Chrome cookie值

    Args:
        encrypted_value: 加密的cookie值
        key: 解密密钥
        meta_version: 数据库版本

    Returns:
        str: 解密后的cookie值
    """
    # 检查加密值的版本
    if encrypted_value.startswith(b'v10'):
        # v10版本使用AES-GCM解密
        encrypted_value = encrypted_value[3:]  # 移除版本前缀

        # 提取nonce、ciphertext和认证标签
        nonce_length = 12  # 96 bits / 8
        nonce = encrypted_value[:nonce_length]
        auth_tag = encrypted_value[-16:]  # AES-GCM认证标签长度为16字节
        ciphertext = encrypted_value[nonce_length:-16]

        try:
            # 尝试导入pycryptodome库进行AES-GCM解密
            try:
                from Crypto.Cipher import AES

                cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
                plaintext = cipher.decrypt_and_verify(ciphertext, auth_tag)

                # 如果meta_version >= 24，移除哈希前缀
                if meta_version >= 24 and len(plaintext) > 32:
                    plaintext = plaintext[32:]

                return plaintext.decode('utf-8')
            except ImportError:
                # 如果没有安装pycryptodome，尝试其他方法
                print("警告: 未安装pycryptodome库，无法解密AES-GCM加密的cookie")
                # 尝试使用DPAPI解密（对于某些情况可能有效）
                raise ImportError("需要pycryptodome库来解密v10版本的cookie")
        except Exception:
            # 尝试使用DPAPI解密（对于旧版本或特殊情况）
            try:
                return decrypt_windows_dpapi(encrypted_value).decode('utf-8')
            except Exception:
                return None
    else:
        # 非v10版本直接使用DPAPI解密
        try:
            return decrypt_windows_dpapi(encrypted_value).decode('utf-8')
        except Exception:
            return None


def main(url= None):
    cookies = get_chrome_cookies_for_url(url)

    if cookies:
        print(f"为 {url} 找到 {len(cookies)} 个cookie:")
        # for name, value in cookies.items():
        #     print(f"{name}: {value}")
        return cookies
    else:
        print(f"未找到 {url} 的cookie")
        return {}

def get_eastmoney_cookie():
    cokkie = main(url="https://www.eastmoney.com")
    if cokkie:
        cookie_pairs = [f"{k}={v}" for k, v in cokkie.items()]
        cookie_pairs = cookie_pairs[::-1]
        cookie_str = '; '.join(cookie_pairs)
        return cookie_str
    else:
        return ""


if __name__ == '__main__':
    url = "https://www.eastmoney.com"
    main(url)

