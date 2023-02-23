import sys
def error_message_detail(error):
    exc_type, exc_value, exc_tb = sys.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename if exc_tb else None
    line_number = exc_tb.tb_lineno if exc_tb else None
    error_message = "Error occurred python script name [{0}] line number [{1}] error message [{2}]".format(file_name, line_number, str(error))
    return error_message



class FlightFareException(Exception):

    def __init__(self,error_message):
        self.error_message = error_message_detail(
            error_message)

    def __str__(self):
        return self.error_message 


