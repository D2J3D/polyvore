from image_processing import process_image

with open('example.jpg', 'rb') as f:
    encoded_image = f.read()

# Тестирование
processed_image = process_image(encoded_image, is_training=True, height=224, width=224)
print(processed_image.shape)  # Должно вывести torch.Size([3, 224, 224])
