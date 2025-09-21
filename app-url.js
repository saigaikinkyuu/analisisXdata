// トレーニングデータのJSONをホストしているURLを指定
const TRAINING_DATA_URL = 'https://example.com/training_data.json';
// 実データ（テスト用）のJSONをホストしているURLを指定
const ACTUAL_DATA_URL = 'https://example.com/actual_data.json';

// グローバルスコープでボキャブラリとエンコーダーを定義
let vocabulary = new Set();
let wordToIndex = {};
let maxLength = 0;

function encode(text) {
    // 句読点や特殊文字を削除して小文字に変換
    const cleanedText = text.replace(/[^\w\s]/g, '').toLowerCase();
    const words = cleanedText.split(' ').filter(word => word.length > 0);
    
    const encoded = words.map(word => wordToIndex[word] || 0); // 未知の単語は0
    
    // パディング
    const padded = new Array(maxLength).fill(0);
    encoded.forEach((value, index) => {
        if (index < maxLength) {
            padded[index] = value;
        }
    });
    
    return padded;
}

// モデルの構築とトレーニング
async function trainModel(trainingData) {
    if (trainingData.length === 0) {
        console.error('トレーニングデータがありません。');
        return null;
    }

    // ボキャブラリを構築し、テキストを数値化する
    trainingData.forEach(data => {
        const cleanedText = data.text.replace(/[^\w\s]/g, '').toLowerCase();
        cleanedText.split(' ').forEach(word => {
            if (word.length > 0) {
                vocabulary.add(word);
            }
        });
    });
    let index = 1;
    vocabulary.forEach(word => {
        wordToIndex[word] = index++;
    });
    
    maxLength = Math.max(...trainingData.map(d => d.text.split(' ').filter(word => word.length > 0).length));
    
    const xs = tf.tensor2d(trainingData.map(data => encode(data.text)));
    const ys = tf.tensor2d(trainingData.map(data => [data.label]));

    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 10, activation: 'relu', inputShape: [maxLength] }));
    model.add(tf.layers.dense({ units: 1 }));

    model.compile({ optimizer: 'adam', loss: 'meanSquaredError' });

    console.log('モデルのトレーニングを開始...');
    await model.fit(xs, ys, {
        epochs: 50,
        batchSize: 2,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                console.log(`Epoch ${epoch + 1}: loss = ${logs.loss.toFixed(4)}`);
            }
        }
    });
    console.log('モデルのトレーニングが完了しました。');
    return model;
}

// JSONデータを取得し、モデルを学習・予測してチャート化
async function run() {
    try {
        console.log('トレーニングデータを取得中...');
        const trainingResponse = await fetch(TRAINING_DATA_URL);
        const trainingJsonData = await trainingResponse.json();
        
        const processedTrainingData = trainingJsonData.map(item => ({
            text: item.message,
            label: item.waitTime
        }));

        // ボキャブラリ構築とモデルの学習
        const model = await trainModel(processedTrainingData);

        if (!model) {
            return;
        }

        console.log('実データを取得し、予測を実行中...');
        const actualResponse = await fetch(ACTUAL_DATA_URL);
        const actualJsonData = await actualResponse.json();
        
        const actualData = actualJsonData.map(item => ({
            text: item.message,
            label: item.waitTime
        }));

        const predictedWaitTimes = [];
        const actualWaitTimes = [];
        for (const item of actualData) {
            const encodedMessage = encode(item.text);
            const prediction = model.predict(tf.tensor2d([encodedMessage]));
            predictedWaitTimes.push(Math.round(prediction.dataSync()[0]));
            actualWaitTimes.push(item.label);
        }

        console.log("実データの実際の待ち時間:", actualWaitTimes);
        console.log("実データの予測された待ち時間:", predictedWaitTimes);

        // 平均値の計算とチャート化
        const averageActualWaitTime = actualWaitTimes.reduce((sum, val) => sum + val, 0) / actualWaitTimes.length;
        const averagePredictedWaitTime = predictedWaitTimes.reduce((sum, val) => sum + val, 0) / predictedWaitTimes.length;

        const ctx = document.getElementById('myChart').getContext('2d');
        if (window.myChartInstance) {
            window.myChartInstance.destroy();
        }

        window.myChartInstance = new Chart(ctx, {
            type: 'line',
            data: {
                labels: actualData.map(item => `ID: ${item.id || actualData.indexOf(item)}`),
                datasets: [{
                    label: '実データの実際の待ち時間',
                    data: actualWaitTimes,
                    borderColor: 'rgba(75, 192, 192, 1)',
                    borderWidth: 2,
                    fill: false,
                    tension: 0.1
                }, {
                    label: '実データの予測待ち時間',
                    data: predictedWaitTimes,
                    borderColor: 'rgba(255, 99, 132, 1)',
                    borderWidth: 2,
                    fill: false,
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: { display: true, text: '待ち時間 (分)' }
                    },
                    x: {
                        title: { display: true, text: 'データポイント' }
                    }
                },
                plugins: {
                    annotation: {
                        annotations: {
                            line1: {
                                type: 'line',
                                yMin: averageActualWaitTime,
                                yMax: averageActualWaitTime,
                                borderColor: 'rgb(75, 192, 192)',
                                borderWidth: 2,
                                borderDash: [6, 6],
                                label: {
                                    enabled: true,
                                    content: `実データの平均値: ${averageActualWaitTime.toFixed(2)}`,
                                    position: 'end'
                                }
                            },
                            line2: {
                                type: 'line',
                                yMin: averagePredictedWaitTime,
                                yMax: averagePredictedWaitTime,
                                borderColor: 'rgb(255, 99, 132)',
                                borderWidth: 2,
                                borderDash: [6, 6],
                                label: {
                                    enabled: true,
                                    content: `予測平均値: ${averagePredictedWaitTime.toFixed(2)}`,
                                    position: 'start'
                                }
                            }
                        }
                    }
                }
            }
        });

    } catch (error) {
        console.error('データの取得またはモデルのトレーニング中にエラーが発生しました:', error);
    }
}

run();