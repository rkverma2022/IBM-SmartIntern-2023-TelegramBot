import telebot
import numpy as np
from waterQuality import WQ
from TestingModel import predict

token = "5999448236:AAGTuwdtA-8VJU8DXLblxxL3MDHlLLZlJDY"
bot = telebot.TeleBot(token=token)


@bot.message_handler(commands=["start"])
def start(message):
    bot.reply_to(message, '''
    
    WELCOME TO Public Welfare and WATER QUALITY PREDICTOR
    
    /help
    
    I AM DELIGHTED TO ASSIST YOU
    
    I usually process the following 9 parameters
    
    1. 'ph'
    2. 'Hardness'
    3. 'Solids'
    4. 'Chlorinates'
    5. 'Sulfate'
    6. 'Conductivity',
    7. 'Organic_carbon'
    8. 'Anthropometries'
    9. 'Turbidity'

     MY COMMANDS: 

     1. /start
     
     2. /help 
     
     3. /checkwq
     
     
     ''')

@bot.message_handler(commands=["help"])
def help(message):
    bot.reply_to(message, '''
                 Chat with me
                 
                 /chat
                 
                 Water Quality
                 
                 /checkwq
                 
    I usually process the following 9 parameters
    
    
    1. ph: pH of 1. water (0 to 14).
    2. Hardness: Capacity of water to precipitate soap in mg/L.
    3. Solids: Total dissolved solids in ppm.
    4. Chloramines: Amount of Chloramines in ppm.
    5. Sulfate: Amount of Sulfates dissolved in mg/L.
    6. Conductivity: Electrical conductivity of water in μS/cm.
    7. Organic_carbon: Amount of organic carbon in ppm.
    8. Trihalomethanes: Amount of Trihalomethanes in μg/L.
    9. Turbidity: Measure of light emiting property of water in NTU.
    

    Instructions : 
    
    1. Please Don't put any space, use comma to separate the parameters values
    2. Please Provide all 9 parameters
    
    Ex: 
    9,145,13168,9,310,592,8,77.2,3.8
    
    
    Thank You !
    
    ''')

    

@bot.message_handler(commands=["checkwq"])
def checkwq(message):
    print(message.text)
    
    global wqAnswer, helpingSentence
    try:
        mess = str(message.text)
        mess = mess.split(" ")[1]
        mess = mess.split(',')
        mess = np.array(mess, dtype=float)
        wqAnswer = ""
        if mess.shape[0] == 9:
            helpingSentence = ""

            wqAnswer = WQ(mess)

        elif mess.shape[0] < 9:
            helpingSentence = "Some Parameters are Missing"

        else:
            helpingSentence = "Remove unnecessary extra value"

        print(f"{mess}  Type {type(mess)} dtype {mess.dtype} {mess.shape}")
    except:
        wqAnswer = "Sorry I can't Process This"
        helpingSentence = "Thank You"

    bot.reply_to(message, wqAnswer + "\n\n" + helpingSentence)

@bot.message_handler()
def chat(message):
    global reply
    try:
        mess = message.text
        reply = predict(mess)
        
    except:
        reply = "Pardon, I can't  Process this query"
    
    bot.reply_to(message, reply)

bot.polling()
