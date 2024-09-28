---
title: Word embedding

---

## Word embedding

### Skip-gram
**Penjelasan Model Skip-Gram dan One-Hot Encoding:**

*Dalam arsitektur skip-gram word2vec* , inputnya adalah kata tengah dan prediksinya adalah kata konteks. Pertimbangkan array kata W, jika W(i) adalah input (kata tengah), maka W(i-2), W(i-1), W(i+1), dan W(i+2) adalah kata konteks jika ukuran jendela geser adalah 2.
![image](https://hackmd.io/_uploads/S1zVrIMCC.png)
adalah diagram yang disajikan dalam paper Word2Vec asli. Pada dasarnya diagram ini menjelaskan bahwa model tersebut menggunakan jaringan saraf dengan satu lapisan tersembunyi(proyeksi)untuk memprediksi kata konteks dengan benar.
Dengan kata lain, model tersebut mencoba memaksimalkan probabilitas mengamati keempat kata konteks secara bersamaan, jika diberikan kata tengah. Secara matematis, hal ini dapat dilambangkan sebagai eq-1 

seperti contoh di bawah ini! :
![image](https://hackmd.io/_uploads/H1kHGUzCC.png)

#### **1. One-Hot Encoding**

*One-hot encoding* adalah metode untuk merepresentasikan kata-kata dalam bentuk vektor biner. Misalnya, dalam sebuah kosakata yang terdiri dari `V` kata unik, kita membuat vektor dengan panjang `V`. Untuk kata tertentu, kita mengatur satu posisi dalam vektor menjadi `1` (yang merepresentasikan kata tersebut), dan posisi lainnya diatur menjadi `0`.

Sebagai contoh, misalnya kita punya kosakata berikut: 

- Kosakata: ["saya", "mahasiswa", "universitas", "trunojoyo", "madura"]

Setiap kata direpresentasikan sebagai vektor satu-hot (one-hot vector) seperti berikut:

- "saya"        → `[1, 0, 0, 0, 0]`
- "mahasiswa"   → `[0, 1, 0, 0, 0]`
- "universitas" → `[0, 0, 1, 0, 0]`
- "trunojoyo"   → `[0, 0, 0, 1, 0]`
- "madura"      → `[0, 0, 0, 0, 1]`

**Rumus One-Hot Encoding**:

Jika `x` adalah kata yang direpresentasikan, maka vektor one-hot untuk `x` di kosakata V dapat direpresentasikan sebagai:
$$x = [0, 0, \dots, 1, \dots, 0] \quad (\text{panjang} = V)
$$

Dimana hanya ada satu `1` yang menandakan posisi kata `x`.

---

#### **2. Model Skip-Gram**

Skip-Gram adalah model pembelajaran kata yang menggunakan neural network sederhana untuk memprediksi kata-kata konteks di sekitar kata pusat. Tujuannya adalah memaksimalkan probabilitas kata-kata konteks yang muncul di sekitar kata pusat. Skip-Gram berfungsi dengan memprediksi konteks berdasarkan kata pusat.

**2.1. Arsitektur Model:**

- **Input Layer**: Merupakan vektor one-hot dari kata pusat dengan dimensi `V` (jumlah kosakata).
- **Hidden Layer**: Merupakan lapisan tersembunyi dengan dimensi `N` (dimensi embedding kata). Input one-hot dikalikan dengan matriks bobot `W_in` (V x N) untuk menghasilkan vektor tersembunyi `h`.
  
  Rumus:
$$h = W_{\text{input}}^T \cdot x
$$
  Dimana:
  - `h` adalah vektor hasil di hidden layer (N-dim).
  - `W_{\text{input}}` adalah matriks bobot input (V x N).
  - `x` adalah vektor one-hot kata pusat (V-dim).

- **Output Layer**: Menghasilkan distribusi probabilitas dari semua kata dalam kosakata menggunakan fungsi **Softmax**. 

  Softmax digunakan untuk menghasilkan probabilitas setiap kata dalam kosakata, berdasarkan konteks kata pusat.

  Rumus Softmax:
$$p(w_{\text{context}} \mid w_{\text{center}}) = \frac{\exp(W_{\text{output}}(w_{\text{context}}) \cdot h)}{\sum_{i=1}^{V} \exp(W_{\text{output}}(i) \cdot h)}
$$
  Dimana:
  - `W_{\text{output}}` adalah matriks bobot output (N x V).
  - `h` adalah vektor tersembunyi dari kata pusat (N-dim).
  - `w_{\text{context}}` adalah kata konteks.
  - `V` adalah jumlah kata dalam kosakata.

  Output softmax adalah probabilitas dari setiap kata dalam kosakata muncul sebagai konteks, mengingat kata pusat tertentu.

---

**2.2. Training Skip-Gram:**

**Tujuan Utama**: Optimalkan representasi vektor kata (embedding) sedemikian rupa sehingga prediksi kata konteks menjadi lebih akurat.

**Fungsi Cost (Objective Function)**:
Model Skip-Gram mencoba memaksimalkan probabilitas semua kata konteks yang benar diberikan kata pusat `w_{t}`, dengan tujuan meminimalkan kesalahan prediksi.

Rumus fungsi cost untuk model Skip-Gram adalah:
$$
J(\theta) = -\frac{1}{T} \sum^T_{t=1} \sum_{-c\leq j \leq c,j\neq 0} \log p(w_{t+j} \mid w_t ;\, \theta)
$$

Dimana:
- `T` adalah jumlah total kata dalam corpus.
- `c` adalah ukuran window (jumlah kata di sekitar kata pusat).
- `w_{t}` adalah kata pusat pada posisi `t`.
- `w_{t+j}` adalah kata konteks pada posisi `t+j`.
- `p(w_{t+j} | w_t; \theta)` adalah probabilitas prediksi konteks yang benar.

**Perluasan Rumus dengan Softmax**:
Rumus ini dapat diperluas dengan memasukkan fungsi softmax di dalamnya:
$$J(\theta) = - \sum_{-c \leq j \leq c, j \neq 0} \left( W_{\text{output}}(w_{t+j}) \cdot h + C \cdot \log \sum_{i=1}^{V} \exp(W_{\text{output}}(i) \cdot h) \right)
$$

---

#### **3. Proses Forward Propagation**

Proses ini adalah bagaimana model memproses data input (kata pusat) dan menghasilkan output (prediksi kata konteks):
- **Langkah 1**: Input berupa vektor one-hot dari kata pusat.
- **Langkah 2**: Vektor input dikalikan dengan matriks bobot `W_input` untuk mendapatkan representasi tersembunyi `h`.
- **Langkah 3**: Hasil `h` dikalikan dengan matriks bobot output `W_output` dan kemudian diterapkan fungsi softmax untuk menghasilkan probabilitas prediksi.

---

#### **4. Backpropagation dan Update Bobot**

Setelah forward propagation selesai, kita melakukan backpropagation untuk menghitung **error** dan memperbarui bobot `W_input` dan `W_output`. Kesalahan prediksi dihitung sebagai selisih antara probabilitas yang diprediksi dengan probabilitas yang benar (y_true):
$$\text{Error} = y_{\text{pred}} - y_{\text{true}}
$$

**Update Bobot dengan Gradient Descent**:
Bobot diperbarui dengan aturan berikut:
$$W_{\text{new}} = W_{\text{old}} - \eta \cdot \nabla J(W)
$$
na:
- `\eta` adalah learning rate.
- `\nabla J(W)` adalah gradien dari fungsi cost terhadap bobot.

Untuk matriks bobot input `W_input` dan output `W_output`, rumus gradiennya adalah sebagai berikut:
$$\nabla W_{\text{input}} = x \cdot \left(W_{\text{output}}^T \cdot \sum_{c=1}^{C} e_c \right)
$$
$$\nabla W_{\text{output}} = h \cdot \sum_{c=1}^{C} e_c
$$

Disini `e_c` adalah error untuk kata konteks `c` dalam window, dan `x` adalah vektor one-hot dari kata pusat.

---

#### **5. Training Skip-Gram dengan Iterasi**

Training melibatkan melakukan forward propagation dan backpropagation dalam banyak iterasi, untuk meminimalkan kesalahan prediksi.

**Contoh Training dengan 2 Iterasi:**
- **Iterasi 1**:
   - Forward propagation untuk kata pusat "passes", prediksi kata konteks "who" dan "the".
   - Backpropagation untuk menghitung error dan memperbarui bobot.
- **Iterasi 2**:
   - Forward propagation untuk kata pusat "man", prediksi kata konteks "the" dan "who".
   - Backpropagation untuk memperbarui bobot.

Setelah dua iterasi, bobot di matriks `W_input` dan `W_output` telah diperbarui dua kali, sehingga model semakin baik dalam memprediksi kata-kata konteks di sekitar kata pusat.

--- 

#### ***Representasi kata dalam vektor dengan 3 elemen untuk setiap kata:***

---

##### **Vektor One-Hot Encoding (3 Elemen)**

Kosakata yang digunakan tetap sama: 
- **Saya**
- **mahasiswa**
- **teknik**
- **informatika**

Namun, setiap kata sekarang direpresentasikan dengan **vektor 3 elemen**.

| Kata         | Vektor One-Hot  (3 Elemen) |
|--------------|----------------------------|
| Saya         | [1, 0, 0]                  |
| mahasiswa    | [0, 1, 0]                  |
| teknik       | [0, 0, 1]                  |
| informatika  | [1, 1, 0]                  |

Catatan: Dalam kasus ini, saya memodifikasi representasi "informatika" agar sesuai dengan dimensi 3.

---

## **Training Samples dengan One-Hot Encoding (3 Elemen)**

Setelah membuat vektor one-hot dengan 3 elemen, setiap kalimat dari **training samples** akan direpresentasikan sebagai urutan vektor one-hot encoding dengan dimensi 3. Berikut ini adalah representasi 3 elemen untuk setiap kalimat:

1. **Training Sample 1**: "Saya mahasiswa teknik informatika"
   - Vektor: `[[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]]`

2. **Training Sample 2**: "Saya mahasiswa"
   - Vektor: `[[1, 0, 0], [0, 1, 0]]`

3. **Training Sample 3**: "Mahasiswa teknik"
   - Vektor: `[[0, 1, 0], [0, 0, 1]]`

4. **Training Sample 4**: "Teknik informatika"
   - Vektor: `[[0, 0, 1], [1, 1, 0]]`

---

##### **Ringkasan dengan 3 Elemen**:

| Training Sample                   | Vektor One-Hot (3 Elemen)                        |
|-----------------------------------|---------------------------------------------------|
| "Saya mahasiswa teknik informatika"| `[[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]]`   |
| "Saya mahasiswa"                  | `[[1, 0, 0], [0, 1, 0]]`                         |
| "Mahasiswa teknik"                | `[[0, 1, 0], [0, 0, 1]]`                         |
| "Teknik informatika"              | `[[0, 0, 1], [1, 1, 0]]`                         |

---

---
## **Perhitungan Manual Model Skip-Gram dengan One-Hot Encoding**

Sebagai contoh, kita akan menggunakan **kosakata** berikut:

- **Kosakata**: ["saya", "mahasiswa", "teknik", "informatika"]
- **One-Hot Encoding** dari kosakata:

| Kata         | Vektor One-Hot |
|--------------|----------------|
| "saya"       | `[1, 0, 0, 0]` |
| "mahasiswa"  | `[0, 1, 0, 0]` |
| "teknik"     | `[0, 0, 1, 0]` |
| "informatika"| `[0, 0, 0, 1]` |

---

#### **Langkah 1: Tentukan kata pusat dan kata konteks**

Misalkan kalimatnya adalah "saya mahasiswa teknik informatika", dan kita akan menggunakan jendela konteks dengan ukuran 1. Jadi, kita akan memprediksi kata konteks di sekitar setiap kata pusat.

- Kata pusat: **"mahasiswa"**  
  Konteks: **"saya"**, **"teknik"**

- Kata pusat: **"teknik"**  
  Konteks: **"mahasiswa"**, **"informatika"**

---

#### **Langkah 2: Forward Propagation (Lanjutan)**
##### **Kata pusat: "mahasiswa"**

- **Input**: Vektor one-hot dari kata "mahasiswa" adalah `[0, 1, 0, 0]`.

1. **Hitung Hidden Layer**  
   Seperti yang sudah dihitung sebelumnya, kita kalikan vektor one-hot `[0, 1, 0, 0]` dengan matriks bobot `W_{\text{input}}`:

   $$
   W_{\text{input}} = \begin{pmatrix}
   0.1 & 0.3 \\
   0.4 & 0.7 \\
   0.5 & 0.2 \\
   0.6 & 0.8
   \end{pmatrix}
   $$
   
   Kalikan dengan vektor one-hot "mahasiswa":
   $$
   h = [0, 1, 0, 0] \cdot W_{\text{input}} = [0.4, 0.7]
   $$
   Jadi, vektor tersembunyi adalah `h = [0.4, 0.7]`.

2. **Hitung Output Layer**  
   Sekarang, kalikan hasil dari hidden layer `[0.4, 0.7]` dengan matriks bobot `W_{\text{output}}`:

   $$
   W_{\text{output}} = \begin{pmatrix}
   0.2 & 0.4 & 0.1 & 0.3 \\
   0.5 & 0.6 & 0.3 & 0.2
   \end{pmatrix}
   $$

   Kalikan dengan hidden layer `[0.4, 0.7]`:
   $$
   z = [0.4, 0.7] \cdot W_{\text{output}} = [0.2 \cdot 0.4 + 0.5 \cdot 0.7, 0.4 \cdot 0.4 + 0.6 \cdot 0.7, 0.1 \cdot 0.4 + 0.3 \cdot 0.7, 0.3 \cdot 0.4 + 0.2 \cdot 0.7]
   $$
   $$
   z = [0.38, 0.64, 0.29, 0.34]
   $$

3. **Softmax Function**  
   Kita gunakan fungsi **softmax** untuk mengonversi hasil `z` menjadi probabilitas:

   $$
   p(y_i) = \frac{\exp(z_i)}{\sum_{k=1}^{V} \exp(z_k)}
   $$
   Hitung eksponensial dari setiap elemen `z`:
   $$
   \exp(z) = [\exp(0.38), \exp(0.64), \exp(0.29), \exp(0.34)] = [1.46, 1.90, 1.34, 1.40]
   $$
   Jumlahkan semua nilai eksponensial:
   $$
   \sum \exp(z) = 1.46 + 1.90 + 1.34 + 1.40 = 6.10
   $$

   Sekarang hitung probabilitas untuk setiap kata dalam kosakata:

   $$
   p(y_1) = \frac{1.46}{6.10} \approx 0.24
   $$
   $$
   p(y_2) = \frac{1.90}{6.10} \approx 0.31
   $$
   $$
   p(y_3) = \frac{1.34}{6.10} \approx 0.22
   $$
   $$
   p(y_4) = \frac{1.40}{6.10} \approx 0.23
   $$

#### **Output: Probabilitas Prediksi**
Probabilitas prediksi dari model Skip-Gram untuk kata pusat "mahasiswa" adalah:

- Probabilitas "saya" sebagai kata konteks: **24%**
- Probabilitas "mahasiswa" sebagai kata konteks: **31%**
- Probabilitas "teknik" sebagai kata konteks: **22%**
- Probabilitas "informatika" sebagai kata konteks: **23%**

Ini adalah output yang dihasilkan dari forward propagation berdasarkan kata pusat "mahasiswa".