git add titanic/image/image.png
git commit
git push

# from my code


## 변수 간 상관관계 파악

>![상관관계](https://github.com/Luna-luv/Kaggle_Study/blob/main/titanic/image/image.png?raw=true)
>
> 너다나비 할 때 깨달은 점인데, 여태까지는 열이 많은 데이터셋을 많이 다루어보지 않아서 변수간 상관관계를 한 눈에 파악할 수 있는 히트맵을 잘 사용하지 않았다. 그런데, 최근 변수가 많은 데이터셋을 많이 접하게 되면서 히트맵의 중요성을 알게 되었다.

---

## seaborn.pairplot을 통한 변수 간 관계 시각화

<p align="center">
<pre><code class="language-python">
g = sns.pairplot(train[[u'Survived', u'Pclass', u'Sex', u'Age', u'Parch', u'Fare', u'Embarked',
       u'FamilySize', u'Title']], hue='Survived', palette = 'seismic',size=1.2,diag_kind = 'kde',diag_kws=dict(shade=True),plot_kws=dict(s=10) )
g.set(xticklabels=[])
</code></pre>
</p>

> 1 pairplot() : 여러 수치형 변수 간 산점도, 분포도, 관계 시각화
> 2 hue = 변수명 : 생존 여부에 따라 색 구분
> 3 diag_king = 'kde' : diag_king(대각선 diagnol kind을 뜻하고, kde는 변수 밀도곡선을 의미) -> 대각선에는 밀도 추정 곡선 표시한다는 것
> 4 palette = 'seismic' : 생존/사망 색상을 seismic 컬러맵으로 설정
> 5 plot_kws=dict(s=10) : 산점도의 점 크기 조절
> 6 g.set(xticklabels=[]) x축 눈금 제거(깔끔하게 보이기 위해)
>
>![pairplot](https://github.com/Luna-luv/Kaggle_Study/blob/main/titanic/image/image-1.png?raw=true)

---

## 모델 특성 정리

> ### Random Forest
> - 설명 : 여러 개의 decision tree를 무작위 feature와 데이터 샘플로 학습한 후 결과를 투표 or 평균해서 예측 
> - 장점 : 과적합에 강하고, 기본 성능이 안정적
> - 주요 파라미터 : n_estimators(트리개수), max_depth(트리 최대 깊이), max_features(각 트리에서 고려할 feature 수), min_samples_split(리프노드가 되기 위한 최소 샘플 수), bootstrap(부트스트랩 샘플링 사용 여부)

> ### Extra Trees(Extremely Randomized Trees)
> - 설명 : 노드 분할 시 가장 좋은 분할을 찾는 대신 **완전히 무작위**로 나눔
> - 장점 : Random Forest보다 더 빠르고 더 높은 편향, 더 낮은 분산
> - 주요 파라미터 : Random Forest와 동일. 단, splitter가 무작위

> ### AdaBoost
> - 설명 : 약한 모델(depth=1 의 tree)을 순차적으로 학습 후 이전 모델이 틀린 샘플에 가중치 부여
> - 주요 파라미터 : n_estimators(약한 모델 개수), learning_rate(각 모델 기여도 조절), base_estimator(약한 모델; 기본은 DecisionTreeClassifier)

> ### Gradient Boosting
> - 설명 : AdaBoost 처럼 순차 학습하지만 잔여 오차를 선형적으로 보정하는 방향으로 학습
> - 장점 : 예측력이 강하고 커스터마이징 가능
> - 주요 파라미터 : n_estimators(트리개수), learning_rate(학습률), max_depth(트리 깊이), subsample(데이터를 일부 샘플링하여 학습->과적합 방지), loss(손실함수)

> ### Support Vector Classifier
> - 설명 : 고차원 공간에서 결정 경계를 찾는 모델로, 마진을 최대화하는 방향으로 학습
> - 주요 파라미터 : C(마진 너비 vs 오차 허용 간의 균형), kernel(커널함수; linear, poly, rbf, sigmoid), gamma(rbf/poly 커널의 곡률 제어), degree(poly 커널의 차수)

---

## 모델 간 비교

### Random Forest vs Extra Trees
|항목|Random Forest|Extra Trees|
|---|---|---|
|분할 방식|최적의 기준 찾음|무작위 기준 선택|
|예측 성능|더 낮은 편향|더 높은 편향|
|과적합 위험|낮음|더 낮음|
|속도|느림|빠름|
|특성 중요도|안정적|조금 덜 정교함|


### AdaBoost vs Gradient Boosting
|항목|AdaBoost|Gradient Boosting|
|---|---|---|
|가중치 방식|잘못 예측된 샘플에 가중치 증가|오차(gradient)를 보정|
|성능|단순함, 과적합에 약함|성능 좋고 유연함|
|이상치 민감도|높음|상대적으로 낮음|
|커스터마이징|제한적|손실함수, 서브샘플 등 다양|


### SVC vs Tree 기반 모델
|항목|SVC|Tree 기반 모델|
|---|---|---|
|기본 개념|마진을 최대화하는 결정 경계|결정 트리의 앙상블|
|선형/비선형 처리|커널 트릭으로 비선형 가능|트리 분기로 비선형 처리|
|스케일 영향|민감함 (정규화 필수)|거의 없음|
|고차원 데이터|잘 작동함|차원의 저주에 영향받을 수 있음|
|속도/확장성|데이터 커지면 느려짐|병렬 처리 가능, 확장성 높음|


---

# from members' code

## Seaborn 라이브러리리

### 1. seaborn.catplot
>
>- 범주형 변수(categorical data)를 시각화할 수 있는 **범용 함수**
>- 내부적으로 `stripplot`, `boxplot`, `violinplot`, `barplot`, `pointplot` 등을 선택해서 그림
>- 여러 범주를 비교할 때 **행/열 분할(grid)** 도 가능
>- `FacetGrid` 객체 반환 → 여러 subplot 자동 생성 가능
>
>> #### 주요 파라미터
>>
| 파라미터 | 설명 |
|----------|------|
| `x`, `y` | 축에 사용할 변수 이름 |
| `hue` | 색상으로 나눌 카테고리 변수 |
| `kind` | 사용할 플롯 종류 (`strip`, `swarm`, `box`, `violin`, `bar`, `point`, `count`) |
| `col`, `row` | subplot으로 나눌 기준 변수 |
| `data` | 사용할 데이터프레임 |
| `order`, `hue_order` | x축과 hue 순서 지정 |
| `height` | subplot 높이 (기본값: 5) |
| `aspect` | subplot 가로/세로 비율 (기본값: 1) |
>>
>> #### kind별 기능 비교
>>
| kind 종류 | 설명 |
|-----------|------|
| `'strip'` | 산점도처럼 개별 포인트 보여줌 (중복 많을 땐 겹침) |
| `'swarm'` | 겹치지 않게 퍼뜨린 점 플롯 (많은 데이터 시 느림) |
| `'box'` | 박스플롯: 중앙값, 사분위수 등 통계 정보 표시 |
| `'violin'` | 바이올린 플롯: boxplot + KDE |
| `'bar'` | 평균값 + 신뢰구간을 막대로 표시 |
| `'point'` | 평균값 + 신뢰구간을 선으로 표시 |
| `'count'` | 각 카테고리의 개수를 막대로 표시 (y 생략 시 자동) |
>>
---
>>
>> #### 그래프 예시
>>
>> 1) violin(boxplot + KDE)
>>
>> ![violin](https://github.com/Luna-luv/Kaggle_Study/blob/main/titanic/image/image-4.png?raw=true)
>>
>> 2) point
>>
>> ![point](https://github.com/Luna-luv/Kaggle_Study/blob/main/titanic/image/image-13.png?raw=true)
>>
>> 3) bar
>>
>> ![bar](https://github.com/Luna-luv/Kaggle_Study/blob/main/titanic/image/image-12.png?raw=true)

---

### 2. seaborn.facetgrid

> - `FacetGrid`는 하나 이상의 범주형 변수를 기준으로 데이터를 **그룹별로 나눠 서브플롯**으로 시각화
> - 각 서브플롯은 **동일한 축과 스케일**을 공유하여 **비교 분석**에 효과적이며, `.map()`을 통해 다양한 플롯을 유연하게 적용할 수 있습니다.
>
>> #### 주요 파라미터
>>
| 파라미터        | 설명                                      |
|------------------|-------------------------------------------|
| `data`           | 시각화할 데이터프레임                      |
| `col`            | 서브플롯을 열 방향으로 나눌 변수           |
| `row`            | 서브플롯을 행 방향으로 나눌 변수           |
| `hue`            | 색상을 기준으로 나눌 범주형 변수           |
| `margin_titles`  | 제목을 subplot 안쪽에 넣을지 여부 (기본: `False`) |
| `height`         | 각 subplot의 높이 (기본: `3`)              |
| `aspect`         | subplot의 가로/세로 비율 (기본: `1`)       |
>>
>> #### 자주 사용하는 지원 플롯
>>
| 지원 플롯         | 설명                        |
|--------------------|-----------------------------|
| `sns.histplot`     | 히스토그램 (분포 확인)       |
| `sns.scatterplot`  | 산점도 (관계 시각화)         |
| `sns.barplot`      | 막대그래프 (평균 등 비교용)  |
>>
---
>> #### 그래프 예시
>>
>> 1) hisplot
>>
>>![hisplot](https://github.com/Luna-luv/Kaggle_Study/blob/main/titanic/image/image-9.png?raw=true)
>>
>> 2) pointplot
>>
>>![pointplot](https://github.com/Luna-luv/Kaggle_Study/blob/main/titanic/image/image-10.png?raw=true)
>>
>> 3) barplot
>>
>>![barplot](https://github.com/Luna-luv/Kaggle_Study/blob/main/titanic/image/image-10.png?raw=true)

---

## 라이브러리별 시각화 사용 용도 비교

### `subplot` vs `FacetGrid` vs `catplot` 

| 항목         | `subplot` (matplotlib)                        | `FacetGrid` (seaborn)                                         | `catplot` (seaborn)                                           |
|--------------|-----------------------------------------------|----------------------------------------------------------------|----------------------------------------------------------------|
| 추상화 수준  | 가장 저수준 (Low-level)                      | 중간 수준 (Mid-level)                                         | 고수준 (High-level)                                           |
| 주요 기능    | 여러 그래프 배치용 도화지                    | 범주형 변수로 서브플롯 나누기                                | 범주형 데이터 자동 시각화                                     |
| 특징         | 자유도 높지만 수동 작업 필요                | 유연하고 다양한 map 지원                                     | 한 줄로 빠르게 그릴 수 있음                                  |
| 사용 대상    | matplotlib 사용자                             | Seaborn에서 비교 시각화 하고 싶은 사람                       | 빠르게 범주형 그래프 만들고 싶은 사람                         |
| 대표 함수    | `plt.subplots()`, `plt.subplot()`            | `sns.FacetGrid()` + `.map()`                                  | `sns.catplot(kind=...)`                                       |
| 내부 구현    | 직접 그리드 구성                             | 내부적으로 subplot 사용함                                    | 내부적으로 FacetGrid 사용함                                   |

---

### 언제 어떤걸 사용??

> 서브플롯 직접 구성하고 싶을 때? -> `subplot`
> 그룹별 분포/관계를 유연하게 **비교**하고 싶을 때 -> `FacetGrid`
> 범주형 변수 시각화를 간단히 하고 싶을 때 -> `catplot`

### 관계 구조

| 상위 도구   | 내부적으로 사용하는 도구 |
|-------------|---------------------------|
| `catplot`   | → `FacetGrid`             |
| `FacetGrid` | → `subplot`               |

