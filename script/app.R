  #### LANGKAH PERTAMA ####
  ####    MEMUAT PACKAGE  ####

# Package yang digunakan
library(raster)
library(terra)
library(sf)
library(neuralnet)
library(caret)

#### LANGKAH KEDUA ####
####  MEMUAT DATA  ####

# Memuat data citra dan training area
citra <- rast("data/WorldView.tif")
traning_area <- st_read("data/TrainingSampel.shp")

# Periksa struktur data awal
print(head(traning_area))
print(unique(traning_area$Id))

####   LANGKAH KETIGA   ####
#### PREPROCESSING DATA ####

# Memastikan Field Id adalah faktor/numerik berurutan
traning_area$Id <- as.factor(traning_area$Id)

# Transformasi proyeksi
trainarea_PCS <- st_transform(traning_area, crs(citra))

# Ekstrak nilai spektral dari titik sampel pada citra
nilai_sampel <- extract(citra, trainarea_PCS)

# Tambahkan nilai spektral ke data training
train_data <- cbind(trainarea_PCS, nilai_sampel)
train_data <- train_data[, c("Id", names(citra))]

# Hapus kolom geometry
data_clean <- st_drop_geometry(train_data)

# Periksa data setelah melakukan preprocessing
print(head(data_clean))
print(str(data_clean))

# Split data menggunakan createDataPartition untuk stratifikasi
set.seed(42)
index <- createDataPartition(data_clean$Id, p = 0.8, list = FALSE)
train_datapart <- data_clean[index, ]
test_data <- data_clean[-index, ]

# Normalisasi dengan metode yang lebih robust
preProcess_params <- preProcess(train_datapart[, -1], method = c("center", "scale"))
train_input <- predict(preProcess_params, train_datapart[, -1])
test_input <- predict(preProcess_params, test_data[, -1])


####    LANGKAH KEEMPAT   ####
#### MODEL NEURAL NETWORK ####

# Persiapan data untuk neural network
train <- cbind(train_input, Id = train_datapart$Id)

# Definisikan formula untuk model neural network
formula <- as.formula(paste("Id ~", paste(names(train_input), collapse = " + ")))

# Latih model neural network dengan parameter yang lebih fleksibel
nn_model <- neuralnet(
  formula, 
  data = train, 
  hidden = c(10, 5),  # Lapisan tersembunyi yang lebih besar
  linear.output = FALSE,
  act.fct = "logistic",  # Fungsi aktivasi
  stepmax = 1e6  # Iterasi maksimum yang lebih besar
)

#Visualisasi struktur model nn
plot(nn_model, main = "Arsitektur Neural Network")

# Prediksi pada data testing
prediksi <- compute(nn_model, test_input)
predicted_classes <- apply(prediksi$net.result, 1, which.max)

#### LANGKAH KELIMA ####
#### EVALUASI MODEL ####

# Ganti baris ini:
test_output <- test_data$Id

# Proses confusion matrix
confusion_matrix <- table(Predicted = predicted_classes, Actual = as.numeric(test_output))
print("Confusion Matrix:")
print(confusion_matrix)

# Hitung akurasi
akurasi <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
print(paste("Akurasi:", round(akurasi * 100, 2), "%"))

####   LANGKAH KEENAM   ####
#### PROSES KLASIFIKASI ####

# Ekstrak nilai spektral untuk seluruh data input
wv_matrix <- as.matrix(citra)
valid_mask <- !is.na(wv_matrix)
wv_matrix[!valid_mask] <- 0

# Normalisasi data citra untuk klasifikasi
wv_matrix_norm <- predict(preProcess_params, wv_matrix)

# Klasifikasi untuk seluruh citra
prediksi_raster <- compute(nn_model, wv_matrix_norm)

# Ambil hasil prediksi (kelas dengan probabilitas tertinggi)
predicted_classes_raster <- as.integer(apply(prediksi_raster$net.result, 1, which.max))

# Ubah hasil prediksi menjadi raster dengan ukuran yang sesuai
predicted_raster <- rast(citra)
values(predicted_raster) <- predicted_classes_raster

####     LANGKAH KETUJUH     ####
#### VISUALISASI KLASIFIKASI ####

# Definisikan warna dan label untuk setiap kelas
kelas_warna <- c("blue", "red", "gray", "darkgreen", "lightgreen")
kelas_label <- c("Badan Air", "Permukiman", "Jalan", "Vegetasi Tegakan", "Vegetasi Non Tegakan")

# Tampilkan hasil
plot(predicted_raster[[1]], 
     main = "Hasil Klasifikasi Tutupan Lahan",
     col = kelas_warna,
     legend = FALSE,
     labels = kelas_label)

legend("bottomright", 
       legend = kelas_label, 
       fill = kelas_warna, 
       title = "Kelas Tutupan Lahan", 
       cex = 0.5)


####        LANGKAH KEDELAPAN       ####
#### EXPORT CITRA HASIL KLASIFIKASI ####


# Export hasil klasifikasi menjadi GeoTIFF
writeRaster(predicted_raster, 
            filename = "output/Klasifikasi_Citra.tif", 
            overwrite = TRUE)

# Cetak konfirmasi
cat("Klasifikasi citra telah diekspor\n")