from PIL import Image
from PIL import ImageFilter
from PIL import ImageEnhance

# 'gandalf.jpg' adlı görseli aç ve 'pic_original' olarak kullan
with Image.open('gandalf.jpg') as pic_original:
    # Görüntü ile ilgili bilgileri yazdır
    print('Açık görüntü\nBoyut:', pic_original.size)
    print('format:', pic_original.format)
    print('tür:', pic_original.mode) # Görüntü modunu (renkli veya siyah-beyaz vb.) yazdır
    pic_original.show() # Görüntüyü ekranda göster
    

    # Görüntüyü gri tonlamaya dönüştür (RGB'den L - grayscale'e)
    pic_gray = pic_original.convert('L')  # 'L' mode, siyah-beyaz/grayscale
    pic_gray.save('gray.jpg')  # Yeni resmi kaydet
    # Yeni görüntü ile ilgili bilgileri yazdır
    print('Görüntü oluşturuldu\nBoyut:', pic_gray.size)
    print('format:', pic_gray.format)
    print('tür:', pic_gray.mode)  # 'L' modunda olduğu için siyah-beyaz (grayscale)
    pic_gray.show() # Yeni görüntüyü göster


    pic_blured = pic_original.filter(ImageFilter.BLUR) # Görüntüye bulanıklık filtresi uygula
    pic_blured.save('blured.jpg') # Bulanıklaştırılmış görseli kaydet
    pic_blured.show()  # Bulanıklaştırılmış görseli göster


    pic_up = pic_original.transpose(Image.ROTATE_180) # Görüntüyü 180 derece döndür
    pic_up.save('up.jpg')
    pic_up.show()


    # 1. Ayna görüntüsü.
    pic_mirrow = pic_original.transpose(Image.FLIP_LEFT_RIGHT) # Görüntüyü yatayda (soldan sağa) çevir (ayna görüntüsü)
    pic_mirrow.save('mirror.jpg')
    pic_mirrow.show()


    # 2. Kontrastı artırma
    pic_contrast = ImageEnhance.Contrast(pic_original) 
    pic_contrast = pic_contrast.enhance(1.5) # Kontrastı 1.5 kat artır
    pic_contrast.save('contr.jpg') 
    pic_contrast.show()


class ImageEditor():
    def __init__(self, filename):
        self.filename = filename
        self.original = None
        self.changed = list()


    def open(self):
        try:
            self.original = Image.open(self.filename)
        except:
            print('Dosya bulunamadı!')
        self.original.show()


    def do_left(self):
        rotated = self.original.transpose(Image.FLIP_LEFT_RIGHT)
        self.changed.append(rotated)


         # Düzenlenen resimlerin otomatik adlandırılması
        temp_filename = self.filename.split('.')
        new_filename = temp_filename[0] + str(len(self.changed)) + '.jpg'


        rotated.save(new_filename)


    # Foto editöründefotoğrafını kırpmak
    def do_cropped(self):
        box = (250, 100, 600, 400) #sol, üst, sağ, alt
        cropped = self.original.crop(box)
        self.changed.append(cropped)


         # Düzenlenen resimlerin otomatik adlandırılması
        temp_filename = self.filename.split('.')
        new_filename = temp_filename[0] + str(len(self.changed)) + '.jpg'


        cropped.save(new_filename)


MyImage = ImageEditor('gandalf.jpg')
MyImage.open()


MyImage.do_left()
MyImage.do_cropped()


for im in MyImage.changed:
    im.show()
