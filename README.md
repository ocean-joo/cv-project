## CV-project
#### [2019-2] 컴퓨터 비전 아이폰 인물사진 모드 확장 프로젝트

2019년도 2학기 컴퓨터비전 수업을 들으면서 진행했던 개인 프로젝트입니다.
아이폰의 인물사진모드가 머신러닝을 이용해 서비스를 제공하지만, 피사체가 사람일때만 모드를 사용할 수 있는 한계가 있다고 생각했습니다.
피사체가 음식과 같은 다른 객체일때도 인물사진 모드가 작동할 수 있도록 했습니다.

Mask-RCNN을 이용해서 피사체의 segmentation mask를 찾은 후, 해당하지 않는 부분을 블러링하는 식으로 아웃 포커싱을 따라했습니다.
openCV에서 제공하는 Mask-RCNN을 사용하니 반드시 버전 3.4.5 이상을 사용해주세요.

**[예시]**

##### [원본1]

<img src="/img/image2.jpeg" width="450px" height="450px" title="image2" alt="image2"></img><br/>

##### [아웃풋1]

<img src="/img/out2.png" width="450px" height="450px" title="out2" alt="out2"></img><br/>


##### [원본2]

<img src="/img/image3.jpeg" width="450px" height="450px" title="image3" alt="image3"></img><br/>

##### [아웃풋2]

<img src="/img/out3.png" width="450px" height="450px" title="out3" alt="out3"></img><br/>



##### [원본3]

<img src="/img/image4.jpeg" width="450px" height="450px" title="image4" alt="image4"></img><br/>

##### [아웃풋3]

<img src="/img/out4.png" width="450px" height="450px" title="out4" alt="out4"></img><br/>




