import face_recognition
from PIL import Image, ImageDraw

image_of_luis = face_recognition.load_image_file('./img/groups/known/Luis_Miron.png')
luis_face_encoding = face_recognition.face_encodings(image_of_luis)[0]

#unknown_image = face_recognition.load_image_file('./img/groups/unknown/luis_photo.png')
#unknown_face_encoding = face_recognition.face_encodings(unknown_image)[0]

#results = face_recognition.compare_faces([luis_face_encoding], unknown_face_encoding)



known_face_encodings = [luis_face_encoding]
known_face_names = ["Luis Miron"]

test_image = face_recognition.load_image_file('./img/groups/known/Luis_Miron.png')
face_locations = face_recognition.face_locations(test_image)
face_encodings = face_recognition.face_encodings(test_image, face_locations)
pil_image = Image.fromarray(test_image)
draw = ImageDraw.Draw(pil_image)
for(top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
  matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

  name = "unknown Person"

  if True in matches:
    first_match_index = matches.index(True)
    name = known_face_names[first_match_index]

  draw.rectangle(((left, top), (right, bottom)), outline=(0,0,0))

  text_width, text_height = draw.textsize(name)
  draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0,0,0), outline=(0,0,0))
  draw.text((left + 6, bottom - text_height - 5), name, fill=(255,255,255,255))

del draw

pil_image.show()


