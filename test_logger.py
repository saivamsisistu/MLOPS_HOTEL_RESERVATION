from src.logger import get_logger
from src.custom_exception import CustomException
import sys

logger=get_logger(__name__)
def divide_numbers(num1,num2):
    try:
        res=num1/num2
        logger.info("dividing %s by %s",num1,num2)
        return res
    except Exception as e:
        logger.error("error occurred: %s", e)
        raise CustomException("An error occurred while dividing numbers", sys)
    
if __name__=="__main__":
    try:
        logger.info("starting the tesiting")
        divide_numbers(10, 0)
    except CustomException as e:
        logger.error(str(e))

