# 간랸한 코드 소개 
## 프로젝트 개요
1. yolo - fall, standing, sitting 등 5가지의 자세를 판단.
2. firebase: 위 5가지 자세에 대한 값을 T/F의 데이터로 저장.
   ex) fall : True, sitting : False
3. flase: 촬영 중인 영상 + yolo를 통해 detect된 정보를 website(http://127.0.0.1:5000)에서 표현


## 작동 방법
1. firebase를 사용하기 위해 json 파일의 경로를 알맞게 설정.(동일 폴더에 추가함)
2. templates folder를 현재 파일 경로에 추가함.
3. 아래 커맨드를 수행
'''python
python app.py

## 작동 flow
1. app.py 실행
2. url을 통해 영상 및 detection 정보 접근 가능.
3. 만약, 0.8 precision이상의 자세가 감지 되었을 경우 firebase의 data update
