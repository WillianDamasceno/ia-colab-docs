# Documentação sobre Google Colab e Python

## Características gerais do Python

### O que é Python?

Python é uma linguagem de programação de alto nível, interpretada. Python é conhecida por sua sintaxe simples e legível, por isso é usada tanto por iniciantes quanto por desenvolvedores mais experientes. Ela suporta diversos paradigmas da programação, dentre entre programação orientada e objetos, funcional e procedural.

### Principais características:

- Sintaxe simples e legível, facilitando escrita e leitura do código.
- Suporta diversos paradigmas de programação.
- Uma grande quantidade de bibliotecas e frameworks, aumentando a produtividade.
- Portabilidade, sendo executável em diversas plataformas.
- Comunidade ativa e documentação completa.

### Prós:

- Facilidade para aprender, por conta de uma sintaxe mais simples, torna o aprendizado mais simples, por isso é uma escolha excelente para iniciantes.
- A grande quantidade de bibliotecas e frameworks ajuda os desenvolvedores a resolverem problemas de forma rápida e eficiente.
- Versatilidade, python pode ser utilizado em diversas áreas, desde desenvolvimento web, cientifico, automações e até machine learning;
- Open Source: python é open source então você pode baixar o código fonte, modificar e usar sua versão personalizada

### Contras:

- Desempenho, por conta do ser uma linguagem interpretada pode ter um desempenho abaixo de outras linguagem por exemplo c++ que é uma linguagem de baixo nível.
- O tamanho do código comparado com outras linguagens pode ser maior para realizar a mesma tarefa.
- Dependências externas, em projetos mais complexos, o gerenciamento de dependências externas pode se tornar complicado;

### Por que usar Machine Learning em Python?

- Ecossistema robusto: por conta de todas os frameworks e bibliotecas populares para machine learning, como TensorFlow, Scikit-learn, Keras e PyTorch, que tornam o desenvolvimento de modelos de machine learning mais eficiente.
- Facilidade de prototipagem: Python é conhecido por sua facilidade de prototipagem, permitindo que os cientistas de dados experimentem rapidamente com diferentes algoritmos e técnicas de machine learning.
- Grande comunidade e recursos: python tem uma comunidade ativa de cientistas de dados e desenvolvedores de machine learning, por conta disso existe uma grande quantidade de recursos disponíveis.
- Integração com outras tecnologias: Python pode ser integrado com outras tecnologias como bando de dados, ferramentas de visualização de dados, e sistemas de produção, tornando-o uma escolha conveniente e poderosa para desenvolvimento de machine learning.

### Fontes:

https://aws.amazon.com/pt/what-is/python/

https://ltecnologia.com.br/blog/vantagens-e-desvantagens-da-linguagem-python/

https://www.linkedin.com/pulse/linguagem-python-3-pr%C3%B3s-e-contras-leia-o-artigo-at%C3%A9-fim-antonio-m--knenf/

## Gerenciador de Pacotes

### O que é o PIP?

O PIP ou "Pip Installs Packages" (Pip Instala Pacotes). É o sistema de gerenciamento de pacotes padrão para Python. Ele permite instalar, atualizar e remover bibliotecas de código Python facilmente.

### Como usar o PIP?

Para instalar um pacote, basta abrir o terminal ou prompt de comando e usar o comando pip install nome_do_pacote. Por exemplo, para instalar o pacote numpy, você digitaria pip install numpy. Para atualizar um pacote, você pode usar pip install --upgrade nome_do_pacote. E para remover um pacote, o comando é pip uninstall nome_do_pacote.

## Bibliotecas Python

Em Python, uma biblioteca (também conhecida como módulo ou pacote) é um conjunto de funçoes, classes e constantes escritas por algum desenvolvedor que podem ser importadas e utilizadas em seu código para realizar tarefas específicas. As bibliotecas são criadas para facilitar o desenvolvimento de software, permitindo que os desenvolvedores reutilizem o código ja existente em vez de escrever tudo do zero.

### 1. Pandas

O pandas é uma biblioteca Python utilizada para manipulação e análise de dados. Ela fornece estruturas de dados poderosas, como DataFrames, e ferramentas para limpeza, transformação e análise de dados.

```python
import pandas as pd

data = {'Nome': ['Alice', 'Bob', 'Charlie'],
        'Idade': [25, 30, 35]}
df = pd.DataFrame(data)

print(df.head())
```

> Resultado:
> | | Nome | Idade |
> |---|---------|-------|
> | 0 | Alice | 25 |
> | 1 | Bob | 30 |
> | 2 | Charlie | 35 |

Neste exemplo:

- Importamos a biblioteca pandas
- É Criando um DataFrame chamado data
- Imprime as primeiras linhas do DataFrame

### 2. NumPy

NumPy é uma biblioteca fundamental para computação científica em python. Ela oferece suporte a arrays multidimensionais e funções matemáticas para operações eficientes.

```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5])

print(np.mean(arr))
```

> Resultado:
>
> ```
> 3.0
> ```

Neste Exemplo:

- Importamos a biblioteca numpy
- É criando um array NumPy
- Mostra na tela a média dos elementos do array

### 3. SciPy

SciPy é uma bibliotec que estende as capacidades do NumPy com funções para otimização, álgebra linear, integração, interpolação, entre outros.

```python
from scipy import optimize

def func(x):
    return x**2 + 5*x + 6

result = optimize.minimize(func, x0=0)
print(result.x)
```

> Resultado:
>
> ```
> [-2.50000002]
> ```

Neste Exemplo:

- Importamos a biblioteca scipy
- Definimos uma função para otimização
- Encontramos o mínimo da função
- Mostramos na tela

### 4. Seaborn

Seaborn é uma biblioteca de visualização de dados baseada na famosa biblioteca matplotlib. Ela simplifica a criação de gráficos estatísticos atrativos e informativos. (Para utilização precisa importar também a matplotlib)

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.scatterplot(x='Idade', y='Nome', data=df)
plt.show()
```

> Resultado:
>
> ![alt text](images/1-seaborn.png)

Neste Exemplo:

- Impostamos as bibliotecas seaborn e matplotlib
- Criamos um gráfico de dispersão com Seaborn
- Mostramos na tela o Gráfico

### 5. Scikit-learn (sklearn)

Scikit-learn é uma biblioteca de aprendizado de máquina que oferece ferramentas simples e eficientes para análise preditiva de dados. Ela inclui algoritmos de classificação, regressão, clustering, entre outros.

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print(predictions)
```

> Resultado:
>
> ```
> [[ 8.3476823 ]
> [ 4.81662104]
> [10.04607379]
> [ 8.21710217]
> [ 8.19395067]
> [ 8.2839789 ]
> [ 6.96622591]
> [ 6.10804438]
> [ 8.94126619]
> [ 7.48852934]
> [ 8.8601208 ]
> [ 5.78851984]
> [ 7.82362983]
> [ 7.60586977]
> [ 6.11154761]
> [ 6.74001543]
> [ 5.89767238]
> [ 4.92517791]
> [ 7.60354103]
> [ 8.83665886]]
> ```

Neste exemplo:

- Importamos as bibliotecas sklearn.model_selection, sklearn.linear_model, sklearn.metrics e numpy
- Os dados de exemplo são gerados usando np.random.rand().
- Os dados são divididos em conjuntos de treinamento e teste usando train_test_split().
- Um modelo de regressão linear é criado usando LinearRegression().
- O modelo é treinado com os dados de treinamento usando fit().
- Previsões são feitas com os dados de teste usando predict().
- As previsões são impressas na tela.
