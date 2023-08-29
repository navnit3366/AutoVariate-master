from py_console import console
class Auto_Var_Logger():

    def __init__(self, typeof):
        super().__init__()
        self.typeof = typeof

    
    def log(self, message):
        if(str(self.typeof).casefold() == "logging"):
            message = "[LOG] " + message
            console.log(message, showTime=True, severe=True)
        elif(str(self.typeof).casefold() == "warning"):
             console.warn(message, showTime=True, severe=True)
        elif(str(self.typeof).casefold() == "error"):
            console.error(message, showTime=True, severe=True)
        elif(str(self.typeof).casefold() == "info"):
            console.info(message, showTime=True, severe=True)
        elif(str(self.typeof).casefold() == "success"):
            console.success(message, showTime=True, severe=True)
      
        
