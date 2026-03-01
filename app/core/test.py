from config import get_settings

settings = get_settings()
print(f"端口是：{settings.app_port}")
print(f"日志级别是：{settings.log_level}")
print(f"应用名是：{settings.app_name}")
