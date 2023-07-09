import logging

from aiogram import Bot, Dispatcher,  executor, types

from utils.loss_and_loaders import image_loader, save_image
from utils.model import NeuralStyleTransfer

# logging level
logging.basicConfig(level=logging.INFO)

# initialize bot
bot = Bot(token='6365644640:AAFODi5jBkVdZ3c79TEEs4CLAaNvbTpKgUU')
dp = Dispatcher(bot)

img_type = ['content_image', 'style_image']
img_idx = 0


# start message
@dp.message_handler(commands=['start'])
async def send_welcome(message: types.Message):
    await bot.send_message(message.chat.id, 'Привет, {0.first_name}!\n'
                                            'Я помогу перенести стиль с одной фотографии на другую 😎'
                                            ''.format(message.from_user),
                           parse_mode='html')

    # keyboard
    markup_general = types.ReplyKeyboardMarkup(resize_keyboard=True)
    button3_general = types.KeyboardButton('Перенести стиль')

    markup_general.add(button3_general)

    await bot.send_message(message.chat.id, 'Нужна помощь? ➡️ /help')
    await bot.send_message(message.chat.id, 'Давай начнем)',
                           parse_mode='html', reply_markup=markup_general)


# help
@dp.message_handler(commands=['help'])
async def help_message(message: types.Message):
    await bot.send_message(message.chat.id, 'Я - бот, который может перенести стиль с одной '
                                            'фотографии на другую. '
                                            'Для этого:\n' 
                                            '1. Нажни на кнопку "Перенести стиль".\n'
                                            '2. Пришли фото, которое нужно преобразовать.\n'
                                            '3. Пришли фото, стиль которой нужно скопировать.\n')


# chat
@dp.message_handler(content_types=['text'])
async def chat(message: types.Message):
    if message.chat.type == 'private':
        if message.text == 'Перенести стиль':
            await bot.send_message(message.chat.id,
                                   'Пожалуйста, отправь мне одним сообщением фотографии в таком порядке:\n'
                                   '1. фото, которое нужно обработать,\n'
                                   '2. фото, стиль которой мы будем повторять.')
        else:
            await bot.send_message(message.chat.id, 'К сожалению, я не понял твою команду(')


# save photo
@dp.message_handler(content_types=['photo'])
async def photo_handler(message: types.Message):
    global img_type
    global img_idx
    photo = message.photo.pop()
    img_idx += 1
    name = img_type[img_idx % 2]
    await photo.download(f'./images/source/{name}.jpg')
    await bot.send_message(message.chat.id, 'Фото успешно загружено.')

    style_reply_markup = types.InlineKeyboardMarkup(row_width=2, one_time_keyboard=True)
    style_reply_button1 = types.InlineKeyboardButton('Продолжить', callback_data='continue')
    style_reply_markup.add(style_reply_button1)

    await bot.send_message(message.chat.id, 'Нажмите "Продолжить"', reply_markup=style_reply_markup)



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

    await bot.send_message(message.chat.id, 'Начинаем обработку фотографии...')

    content_img = image_loader(content_image_name)
    style_img = image_loader(style_image_name)
    input_img = content_img.clone()

    output = nst.run_style_transfer(content_img, style_img, input_img)
    save_image(output, 'images/results/result_img.jpg')

    await bot.send_message(message.chat.id, 'Готово!')

    result = open('images/results/result_img.jpg', 'rb')

    await bot.send_photo(message.chat.id, result)

    await bot.send_message(message.chat.id, 'Чтобы попробовать ещё раз, просто отправь мне новое фото.')


# launch long polling
if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
