import logging
import sys
from config import settings

def setup_logging():
    """
    配置全局日志记录器。
    
    设置日志级别、格式和处理器 (Console)。
    """
    logging.basicConfig(
        level=settings.LOG_LEVEL,
        format=settings.LOG_FORMAT,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # 调整一些第三方库的日志级别，避免刷屏
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("pymilvus").setLevel(logging.WARNING)
    
    logger = logging.getLogger("app")
    logger.info(f"Logging initialized with level: {settings.LOG_LEVEL}")
    return logger

# 创建一个默认的 logger 实例供导入使用
logger = logging.getLogger("app")
