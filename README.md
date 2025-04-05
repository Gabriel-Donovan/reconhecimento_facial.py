import cv2
import os

def detect_faces(image_path):
    """Detecta rostos na imagem e marca com quadrados"""
    # Carrega a imagem
    image = cv2.imread(image_path)
    if image is None:
        print(f"Erro: Não foi possível carregar a imagem em {image_path}")
        return None
    
    # Converte para tons de cinza para a detecção
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Carrega o classificador Haar Cascade
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    # Detecta os rostos na imagem
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30))
    
    if len(faces) == 0:
        print("Nenhum rosto detectado na imagem.")
        return None
    
    # Desenha quadrados vermelhos ao redor de cada rosto detectado
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)  # Vermelho, espessura 2
    
    return image

def main():
    # Caminho da imagem - substitua pelo seu
    image_path = r"C:\Users\navondes\Downloads\teste rosto1.jpg"
    
    if not os.path.exists(image_path):
        print(f"Arquivo não encontrado: {image_path}")
        return
    
    # Detecta e marca os rostos
    result_image = detect_faces(image_path)
    
    if result_image is not None:
        # Mostra o resultado
        cv2.imshow("Rostos Detectados", result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Opcional: Salvar o resultado
        output_path = os.path.join(os.path.dirname(image_path), "rostos_detectados.jpg")
        cv2.imwrite(output_path, result_image)
        print(f"Resultado salvo em: {output_path}")

if __name__ == "__main__":
    main()
