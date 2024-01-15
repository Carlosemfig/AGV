import cv2
import numpy as np

#chamar o video que quermos usar
cap=cv2.VideoCapture(r"C:\Users\hvendas\Desktop\GIT\opticalFlow\steel_ball.mp4")

#criar a classe object detector
#history default=500
#vartreshold default= 16 (diminuir faz com que assuma mais coisas do espaço= pior)
object_detector=cv2.createBackgroundSubtractorMOG2(varThreshold=200)
#object_detector=cv2.createBackgroundSubtractorKNN()


#este é o kernel para CLOSE, quanto maior o tamanho do kernel, mais close. Os numeros do kernel só podem ser numeros impares.
kernel = np.ones((15, 15), np.uint8) 



while True:
    #este é para ler o video que queremos usar
    #ret é true enquanto o video tiver frames. Frame são todos os frames do video
    ret, frame=cap.read()

    #criar uma mask do frame
    mask=object_detector.apply(frame)
    #tira as sombras usando o treshold
    _, mask= cv2.threshold(mask,254,255,cv2.THRESH_BINARY)
    #mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

 
    #encontra os contornos de todos os objectos brancos da imagem e coloca-os numa lista, contours
    contours,_=cv2.findContours(mask, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)


    for cnt in contours:
        #vai buscar cada um dos contornos e calcular a sua área
        area = cv2.contourArea(cnt)
 
        if area > 5000:
            #apenas contornos com area superior a 5000 serão considerados verdadeiros contornos, todos os outros serão descartados com barulho.
            print(area)
            #desenhar os contornos que satisfazem a condição a verde.
            cv2.drawContours(frame, [cnt], -1, (0,255,0),2)
            #desenhar uma bounding box à volta dos contornos a azul.
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 7)

    #faz display de todos os frames, sendo um video in real time
    #faz resize da frame, para fit no meu ecrã
    frame = cv2.resize(frame, (640, 480))
    cv2.imshow("Video",frame)

    #faz display da mask
    mask = cv2.resize(mask, (640, 480))
    cv2.imshow("Mask",mask)


    key = cv2.waitKey(30)
    #clicar no esc para parar o video
    if key==27:
        break

cap.release()
cv2.destroyAllWindows()

