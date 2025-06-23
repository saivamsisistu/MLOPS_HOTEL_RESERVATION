import traceback
import sys

class CustomException(Exception):
    def __init__(self, error_message, error_detail):
        super().__init__(error_message)
        self.error_message = self.get_detailed_error_message(error_message, error_detail)
    
    @staticmethod
    def get_detailed_error_message(error_message, error_detail):
        _, _, exc_tb = sys.exc_info()
        if exc_tb is not None:
            file_name = exc_tb.tb_frame.f_code.co_filename
            line_number = exc_tb.tb_lineno
            error_message = f"{error_message} occurred in file {file_name} at the line number {line_number}"
        return error_message
    # def get_detailed_error_message(error_message,error_detail:sys):
    #     if error_detail is not None:
    #         tb=traceback.extract_tb(error_detail.__traceback__)
    #         if tb:
    #             file_name=tb[-1].filename
    #             line_number=tb[-1].lineno
    #             return f"{error_message} occurred in file {file_name} at the line number {line_number}"
    def __str__(self):
        return self.error_message

    