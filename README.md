## 1. 간 대사 효소의 안정성에 영향을 미치는 요소들

- 온도: 효소는 일반적으로 가장 효율적으로 작동하는 최적의 온도가 존재한다. 너무 높거나 낮은 온도는 효소를 변성시키거나 활성을 줄인다.

- pH 수준: 온도와 마찬가지로 효소에는 최적의 pH 수준이 존재한다. 이 pH에서 벗어나면 효소 활성이 감소할 수 있다.

- 효소 농도: 효소의 농도는 안정성에 영향을 줄 수 있다. 너무 많거나 적으면 부정확한 결과를 초래할 수 있다.

- 기질 농도: 화합물(기질)의 농도도 효소 활성에 영향을 줄 수 있다.

- 공요소 및 억제제: 공요소의 존재는 효소 활성에 필요할 수 있으며, 억제제는 활성을 줄일 수 있다.

- 버퍼 구성: 버퍼의 구성은 효소의 안정성에 영향을 줄 수 있으며, 특히 효소나 기질과 예상치 못한 방식으로 상호작용하는 경우에 그러할 수 있다.

## 2. 화합물의 물성 성질

- 용해도: 반응 매체에서 화합물의 용해도는 효소에 대한 그 가용성에 큰 영향을 줄 수 있다. 불용성 화합물은 효소에 접근할 수 없어 반응 속도가 감소할 수 있다.

- 분자량: 화합물의 크기와 무게는 효소의 활성 부위와 어떻게 상호 작용하는지에 영향을 줄 수 있다.

- 극성: 화합물의 극성은 그 용해도와 효소와의 상호작용에 영향을 줄 수 있다. 극성 화합물은 비극성 화합물과 비교하여 효소와 다르게 상호작용할 수 있다.

- 안정성: 반응 조건에서 화합물의 안정성이 중요하다. 화합물이 불안정하면 분해되거나 형태가 바뀌어 반응에 영향을 줄 수 있다.

- 입체화학: 화합물 내 원자의 공간 배열은 효소에 어떻게 결합하는지에 영향을 줄 수 있다. 또한 이성질체는 다른 활성을 가질 수 있다.

- 반응성: 시스템 내 다른 분자와의 화합물의 본래 반응성은 효소의 활성에 영향을 줄 수 있다. 매우 반응성이 높은 화합물은 효소나 시스템의 다른 구성 요소와 간섭할 수 있다.

- 독성: 화합물이 효소가 존재하는 생물학적 시스템에 독성이 있다면 전반적인 반응에 영향을 줄 수 있다.

- 결합 친화력: 화합물이 효소에 얼마나 강하게 결합하는지는 반응 속도론에 영향을 줄 수 있다.

- LogP(분배 계수): 이 값은 화합물의 지질 친화성을 나타내며, 세포 막을 통과하고 세포 내 효소와 상호작용하는 능력에 영향을 줄 수 있다.

- pKa(산 해리 상수): 이 값이 다른 pH 수준에서 화합물의 전하 상태에 영향을 줄 수 있으며, 효소와의 상호 작용에 영향을 줄 수 있다.

## 3. 전자친화도, 가중 전자친화도, 원소 특징

- 전자친화도(electron_affinities): 전자친화도는 중성 원자가 기체 상태에서 전자를 받아들여 음이온을 형성할 때 방출되는 에너지의 양을 나타낸다. 값이 클수록 전자를 얻으려는 경향이 크다. 예를 들어, 염소(Cl)는 전자친화도가 349.0 kJ/mol로 전자를 쉽게 얻지만, 베릴륨(Be)은 -48 kJ/mol로 전자를 얻기 어렵다.


- 가중 전자친화도(wtd_electron_affinities): 특정 목적을 위해 조정된 전자친화도로, 전자친화도에 다양한 요소의 영향을 고려한 값이다. 여기서 인(P)과 황(S) 같은 원소는 각각 1.2와 1.5의 배수로 전자친화도가 조정되어 있다. 이는 특정 상황에서 이들 원소의 전자 획득 경향을 반영한 것이다.
원소 특성:

- 원소 특성(element_properties): 원자량과 자연 존재 비율이 포함된다. 원자량은 해당 원소의 원자 평균 질량을 의미하며, 자연 존재 비율은 자연에서 해당 원소가 존재하는 상대적인 비율이다. 이러한 값들은 화합물 내 원소의 물리적 및 화학적 행동을 이해하는 데 중요하다.

이 세 가지의 값을 AI에 이용하기 위해 다음과 같은 값으로 치환했다.

```
electron_affinities = {
    'Br':324.6,
    'Se':79.9,
    'se':79.9,
    'I':126.9,
    'H': 72.8,
    'Li': 59.6,
    'Be': -48,
    'B': 26.7,
    'C': 121.8,
    'c': 121.8,
    'N': 7.0,
    'O': 140.4,
    'o': 140.4,
    'F': 328.0,
    'Na': 52.8,
    'Mg': -40,
    'Al': 42.5,
    'Si': 134.1,
    'P': 72.0,
    'S': 200.0,
    's': 200.0,
    'Cl': 349.0
}

wtd_electron_affinities = {
    'Br':324.6,
    'Se':79.9,
    'I':126.9,
    'H': 72.8,
    'Li': 59.6,
    'Be': -48,
    'B': 26.7,
    'C': 121.8,
    'N': 7.0,
    'O': 140.4,
    'F': 328.0,
    'Na': 52.8,
    'Mg': -40,
    'Al': 42.5,
    'Si': 134.1,
    'P': 72.0 * 1.2,
    'S': 200.0 * 1.5,
    'Cl': 349.0
}

element_properties = {
    'Br':{'Atomic_weight': 79.904, 'Natural_abundance': 50.69},
    'Se':{'Atomic_weight': 78.96, 'Natural_abundance': 49.61},
    'se':{'Atomic_weight': 78.96, 'Natural_abundance': 49.61},
    'I':{'Atomic_weight': 126.904, 'Natural_abundance': 100},
    'H': {'Atomic_weight': 1.008, 'Natural_abundance': 99.985},
    'Li': {'Atomic_weight': 6.94, 'Natural_abundance': 92.58},
    'Be': {'Atomic_weight': 9.012, 'Natural_abundance': 100},
    'B': {'Atomic_weight': 10.81, 'Natural_abundance': 80.1},
    'C': {'Atomic_weight': 12.011, 'Natural_abundance': 98.93},
    'c': {'Atomic_weight': 12.011, 'Natural_abundance': 98.93},
    'N': {'Atomic_weight': 14.007, 'Natural_abundance': 99.63},
    'O': {'Atomic_weight': 15.999, 'Natural_abundance': 99.76},
    'o': {'Atomic_weight': 15.999, 'Natural_abundance': 99.76},
    'F': {'Atomic_weight': 18.998, 'Natural_abundance': 100},
    'Na': {'Atomic_weight': 22.990, 'Natural_abundance': 100},
    'Mg': {'Atomic_weight': 24.305, 'Natural_abundance': 78.99},
    'Al': {'Atomic_weight': 26.982, 'Natural_abundance': 100},
    'Si': {'Atomic_weight': 28.085, 'Natural_abundance': 92.23},
    'P': {'Atomic_weight': 30.974, 'Natural_abundance': 100},
    'S': {'Atomic_weight': 32.06, 'Natural_abundance': 95.02},
    's': {'Atomic_weight': 32.06, 'Natural_abundance': 95.02},
    'Cl': {'Atomic_weight': 35.45, 'Natural_abundance': 75.78},
}
```

## 4. VarianceThreshold

VarianceThreshold는 사이킷런 라이브러리에서 제공하는 특성 선택 방법 중 하나로, 주어진 분산 임계값보다 낮은 분산을 가진 특성을 제거하는 방법이다. 이는 데이터의 특성 중 일부가 거의 변하지 않거나 완전히 일정한 경우, 그러한 특성이 모델의 성능 향상에 기여하지 않을 가능성이 높기 때문에 유용하다.

현재 코드에서는 다음과 같은 값으로 이용했다.

```
transform = VarianceThreshold(threshold=0.01)
```


## 5. MorganFingerprint

Rdkit의 AllChem.GetHashedMorganFingerprint를 이용하여 Morgan 지문을 생성할 수 있다. 이 함수의 매개변수를 설정하여 반지름과 비트수를 입력하고 지문의 특성을 결정할 수 있다.

- 반지름: 반지름은 지문이 포착하는 원자의 지역 화학 환경의 범위를 결정한다. 반지름이 작으면 지문은 각 원자 주변의 더 좁은 환경을 포착하고, 반지름이 크면 더 넓은 환경을 포착한다.

- 비트수: 비트 수는 지문 벡터의 길이를 결정한다. 비트 수가 더 크면 지문은 더 많은 정보를 포함할 수 있으며, 더 작으면 더 적은 정보를 포함한다. 비트 수가 너무 적으면 충돌이 발생할 수 있으며, 여러 다른 화학 구조가 동일한 지문을 생성할 수 있다.

이런 분자의 지문을 나타내는 벡터에서 일부 비트는 특정 데이터셋에서 거의 변하지 않을 수 있다. 이러한 비트는 모델에 유용한 정보를 제공하지 않을 가능성이 높으므로 VarianceThreshold을 이용해서 제거하는 것이 좋을 수 있다.

코드에서는 다음과 같은 값으로 이용했다. nBits의 값을 더 줄일수도 있지만, 저 값이 최적으로 결과값이 잘 나왔다.

```
def mol2fp(mol):
    fp = AllChem.GetHashedMorganFingerprint(mol, 8, nBits=4096*2)
    ar = np.zeros((1,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, ar)
    return ar
```

## 6. 대사 안정성을 평가할 때, MorganFingerprint에 부족할 수 있는 점


  1. 역학적 정보 부족: Morgan 지문은 분자의 구조적 측면에 중점을 두며 본질적으로 엔트로피, 엔탈피 또는 자유 에너지와 같은 열역학적 특성에 대한 정보를 포함하지 않는다. 이러한 요소는 안정성을 평가하는 데 중요하다.

- 해결방법

  - 분자의 엔트로피, 자유 에너지


2. 전자 구조 정보 없음: 분자의 안정성을 이해하려면 분자 궤도, 전자 분포 및 결합 강도에 대한 정보를 포함하여 전자 구조에 대한 지식이 필요한 경우가 많다. Morgan 지문은 이러한 수준의 세부 정보를 제공하지 않는다.

- 해결방법
  - 분자의 전자 친화도, Gastiere 전하

Gastiere 전하란?

ㄱ. 분자의 전자적 구조를 고려하지 않고, 각 원자의 전체 분자 환경을 고려하지 않으므로 모든 종류의 원자에 대해 정확하지 않을 수 있다. 특지 전이 금속과 관련된 분자 또는 특이한 전자 상태의 분자이다.

ㄴ. 평형 상태에서 분자 내의 원자가 동일한 전기 음성도를 가져야 한다고 가정하여 분자 전체의 전자 밀도 재분포를 근사화한다.

코드에서는 다음과 같은 방법으로 사용된다.

```
def calculate_gasteiger_charges(molecule):
    AllChem.ComputeGasteigerCharges(molecule)
    charges = [float(atom.GetProp('_GasteigerCharge')) for atom in molecule.GetAtoms()]
    return charges
```

3. 국소 환경으로 제한됨: Morgan 지문의 반경 메개변수를 조정하여 보다 확장된 구조 정보를 캡처할 수 있지만 여전히 기본적으로 각 원자 주변의 국지적 환경에 초점을 맞추고 있다. 전역 속성이나 장거리 상호 작용은 명시적으로 포착되지 않는다.
- 해결방법
  - MACCS Keys: 166개의 이진 비트로 구서오디어 있으며, 분자의 전역적 특성을 일부 포착할 수 있다
  - ECFP(Extended Connectivity Fingerprints): Morgan Fingerprints의 변형이며, 유사한 화학적 구조를 가진 분자를 식별하는 데 효과적이다.
4. 해싱 충돌 가능성: Morgan 지문의 해시된 버전은 다른 구조적 특징이 같은 비트에 매핑될 수 있어 충돌이 발생할 수 있다.

5. 입체 화학적 측면 무시: 분자 안정성은 분자 내 원자의 3D 공간 배열이나 특정 구조에 의존할 수도 있습니다. 기본 Morgan 지문은 3D 구조를 고려하지 않고 2d 구조정보에 중점을 둔다.

- 해결방법
  - 여러 순응자를 생성하고 각 순응자에 대한 설명자를 계산하면 3D 공간 배열의 가변성을 포착하는 데 도움이 될 수 있다.

코드에서는 다음과 같은 방법으로 사용된다.

```
conformers = AllChem.EmbedMultipleConfs(mol, numConfs=10)
for confId in conformers:
    AllChem.MMFFOptimizeMolecule(mol, confId=confId)
```


## 7. 추가적인 feature

이 외에도 전자친화도와 가중 전자친화도와 관련된 계산 feature를 추가했다. 코드에서는 다음과 같은 방법으로 사용된다.

```
def calculate_affinities(mol):
    affinities = []
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        affinity = electron_affinities.get(symbol, None)
        if affinity is not None:
            affinities.append(affinity)
    return affinities

def wtd_calculate_affinities(mol):
    affinities = []
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        affinity = wtd_electron_affinities.get(symbol, None)
        if affinity is not None:
            affinities.append(affinity)
    return affinities

def calculate_range(affinities):
    return np.max(affinities) - np.min(affinities)

def wtd_calculate_average_atomic_mass(mol):
    average_atomic_mass = 0
    total_abundance = 0
    
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        properties = element_properties.get(symbol, None)
        if properties is not None:
            atomic_weight = properties['Atomic_weight']
            natural_abundance = properties['Natural_abundance']

            average_atomic_mass += atomic_weight * (natural_abundance / 100)
            total_abundance += natural_abundance / 100

    if total_abundance != 0:
        average_atomic_mass /= total_abundance
        
    return average_atomic_mass

def make_feat(train):
    PandasTools.AddMoleculeColumnToFrame(train,'SMILES','Molecule')
    train["FPs"] = train.Molecule.apply(mol2fp)

    train["addH"] = train.Molecule.apply(AllChem.AddHs)
    train['Charges'] = train['addH'].apply(calculate_gasteiger_charges)

    train['affinities'] = train['Molecule'].apply(calculate_affinities)
    train['wtd_affinities'] = train['Molecule'].apply(wtd_calculate_affinities)

    train['mean_affinity'] = train['affinities'].apply(np.mean)
    train['gmean_affinity'] = train['affinities'].apply(scipy.stats.gmean)
    train["entropy_affinity"] = train['affinities'].apply(scipy.stats.entropy)
    train['range_affinity'] = train['affinities'].apply(calculate_range)
    train['std_affinity'] = train['affinities'].apply(np.std)

    train['wtd_mean_affinity'] = train['wtd_affinities'].apply(np.mean)
    train['wtd_gmean_affinity'] = train['wtd_affinities'].apply(scipy.stats.gmean)
    train["wtd_entropy_affinity"] = train['wtd_affinities'].apply(scipy.stats.entropy)
    train['wtd_range_affinity'] = train['wtd_affinities'].apply(calculate_range)
    train['wtd_std_affinity'] = train['wtd_affinities'].apply(np.std)

    train['length'] = train['wtd_affinities'].apply(len)
    train = train.drop(['affinities', 'wtd_affinities'], axis = 1)

    train['GasteigerCharges'] = train['addH'].apply(calculate_gasteiger_charges)

    train = train.drop(["addH",'Molecule', 'FPs','Charges', 'GasteigerCharges'], axis = 1)
    
    return train
```



