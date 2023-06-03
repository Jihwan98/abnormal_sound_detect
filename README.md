# abnormal_sound_detect

- 음성을 통한 위험상황 탐지가 왜 필요한가? 카메라만 있으면 되는 것 아닌가?

→ No. 불완전하다. 카메라가 잡을 수 없는 사각지대, 혹은 가스새는소리 등 모션이 없지만 위급상황인 것이 있기 때문에

- 시나리오 : 쉬운 시나리오 부터. 집을 비운 상황에선 어떠한 소리가 나면 abnormal일 것, 독거노인 혼자 지내기 때문에 작은 tv소리나 웅얼거리는 작은 소리는 나지만, 선명한 말소리 큰 소리 등은 abnormal일 것

## 데이터

- Ai hub에서 구축한 위급상황 음성/음향 데이터를 사용.
- 이 중, 일상생활 데이터의 경우가 더 구축이 잘 되어있기 때문에 이 데이터를 활용하여 Auto Encoder를 통한 abnormal sound detection을 해보려고 한다.
- [위급상황 음성/음향](https://aihub.or.kr/aidata/30742)

## 모델
AutoEncoder 모델을 통해서 시작해보았음. loss가 작으면 normal 크면 abnormal

![image](https://github.com/Jihwan98/abnormal_sound_detect/assets/76936390/5c9159ab-f1af-4271-9c4a-2f313012b1c2)

```python
class AE(nn.Module):
    def __init__(self, input_channel=1, h_dim=128*5*10, z_dim=512):
        super(AE, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
        
    
    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        return out
```

## 전처리 방법에 따른 차이
1. raw
2. mel spectogram -> (40, 80)
3. log mel spectogram -> (40, 80)

-> log mel spectogram으로 전처리 했을 때 가장 성능이 좋음.  [논문](https://scienceon.kisti.re.kr/srch/selectPORSrchArticle.do?cn=JAKO202016357498458&dbt=NART)에서도 log mel spectogram이 가장 성능이 좋다고 함

![image](https://github.com/Jihwan98/abnormal_sound_detect/assets/76936390/fd83d8d4-2b3e-47ce-af79-c51dbf5d9119)

## 음성 길이에 따른 차이
1초 vs 2초 => 2초의 경우 성능이 훨씬 좋아짐.

![image](https://github.com/Jihwan98/abnormal_sound_detect/assets/76936390/0df5cef2-e980-4a3d-a029-f0ba51fc4b97)



## 참고 자료

[기계 진동(소음)에 나타나는 이상 패턴을 자동으로 탐지할 수 있을까? (2)](https://inforience.net/2019/06/08/machine-vibration2/)

→ 1D-VAE와 2D-VAE를 해봤을 때, 성능은 비슷하나 1D-VAE가 파라미터가 적어 빨랐다고함. 먼저 1D-VAE로 진행.

[Abnormal Sound Detection and Identification in Surveillance System](https://www.koreascience.or.kr/article/CFKO201021868482452.pub?orgId=kips)
