import logging
import os

from aiogram import Bot, Dispatcher,  executor, types

from utils.loss_and_loaders import image_loader, save_image
from utils.model import NeuralStyleTransfer

# logging level
logging.basicConfig(level=logging.INFO)

# initialize bot
TG_BOT_TOKEN = os.environ.get('TG_BOT_TOKEN')
bot = Bot(token=TG_BOT_TOKEN)
dp = Dispatcher(bot)

img_type = ['content_image', 'style_image']
img_idx = 0


# start message
@dp.message_handler(commands=['start'])
async def send_welcome(message: types.Message):
    await bot.send_message(message.chat.id, 'Hi, {0.first_name}!\n'
                                            'I will help you transfer the style from one photo to another 😎'
                                            ''.format(message.from_user),
                           parse_mode='html')

    # keyboard
    markup_general = types.ReplyKeyboardMarkup(resize_keyboard=True)
    button3_general = types.KeyboardButton('Transfer Style')

    markup_general.add(button3_general)

    await bot.send_message(message.chat.id, 'Need help? ➡️ /help')
    await bot.send_message(message.chat.id, "Let's get started)",
                           parse_mode='html', reply_markup=markup_general)


# help
@dp.message_handler(commands=['help'])
async def help_message(message: types.Message):
    await bot.send_message(message.chat.id, 'I am a bot that can transfer style from one photo to another. '
                                            'For this:\n' 
                                            '1. Click on the "Transfer Style" button.\n'
                                            '2. Send a photo that needs to be converted.\n'
                                            '3. Send a photo, the style of which needs to be copied.\n')


# chat
@dp.message_handler(content_types=['text'])
async def chat(message: types.Message):
    if message.chat.type == 'private':
        if message.text == 'Transfer Style':
            await bot.send_message(message.chat.id,
                                   'Please send me the photos in this order in one message:\n'
                                   '1. photo to be processed,\n'
                                   '2. photo, the style of which we will repeat.')
        else:
            await bot.send_message(message.chat.id, "Unfortunately, I didn't understand your command(")


# save photo
@dp.message_handler(content_types=['photo'])
async def photo_handler(message: types.Message):
    global img_type
    global img_idx
    photo = message.photo.pop()
    img_idx += 1
    name = img_type[img_idx % 2]
    await photo.download(f'./images/source/{name}.jpg')
    await bot.send_message(message.chat.id, 'Photo uploaded successfully.')

    style_reply_markup = types.InlineKeyboardMarkup(row_width=2, one_time_keyboard=True)
    style_reply_button1 = types.InlineKeyboardButton('Continue', callback_data='continue')
    style_reply_markup.add(style_reply_button1)

    await bot.send_message(message.chat.id, 'Click "Continue"', reply_markup=style_reply_markup)



# callback
@dp.callback_query_handler(lambda call: True)
async def callback_inline(call):
    await launch_nst(call.message)


# launch style transfer
async def launch_nst(message):
    print('ok')
    content_image_name = 'images/source/content_image.jpg'
    style_image_name = 'images/source/style_image.jpg'

    nst = NeuralStyleTransfer()

    await bot.send_message(message.chat.id, 'We begin processing the photo...')

    content_img = image_loader(content_image_name)
    style_img = image_loader(style_image_name)
    input_img = content_img.clone()

    output = nst.run_style_transfer(content_img, style_img, input_img)
    save_image(output, 'images/results/result_img.jpg')

    await bot.send_message(message.chat.id, 'Done!')

    result = open('images/results/result_img.jpg', 'rb')

    await bot.send_photo(message.chat.id, result)

    await bot.send_message(message.chat.id, 'To try again, just send me a new pics.')


# launch long polling
if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
