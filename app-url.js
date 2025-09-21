// トレーニングデータのJSONをホストしているURLを指定
const TRAINING_DATA_URL = './json/training_data.json';
// 実データ（テスト用）のJSONをホストしているURLを指定
const ACTUAL_DATA_URL = './json/actual_data.json';

// テキストを数値化するためのボキャブラリとエンコーダー
let vocabulary = new Set();
let wordToIndex = {};
let maxLength = 0;

function encode(text) {
    const words = text.split(' ');
    const encoded = words.map(word => wordToIndex[word] || 0);
    while (encoded.length < maxLength) {
        encoded.push(0);
    }
    return encoded;
}

// モデルの構築とトレーニング
async function trainModel(trainingData) {
    if (trainingData.length === 0) {
        console.error('トレーニングデータがありません。');
        return null;
    }

    // ボキャブラリの構築とテキストの数値化
    trainingData.forEach(data => {
        data.text.split(' ').forEach(word => vocabulary.add(word));
    });
    let index = 1;
    vocabulary.forEach(word => {
        wordToIndex[word] = index++;
    });
    maxLength = Math.max(...trainingData.map(d => d.text.split(' ').length));

    const xs = tf.tensor2d(trainingData.map(data => encode(data.text)));
    const ys = tf.tensor2d(trainingData.map(data => [data.label]));

    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 10, activation: 'relu', inputShape: [xs.shape[1]] }));
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

async function run() {
    try {
        console.log('トレーニングデータを取得中...');
        const trainingResponse = await fetch(TRAINING_DATA_URL);
        const trainingData = await trainingResponse.json();
        const processedTrainingData = trainingData.map(item => ({
            text: item.message,
            label: item.waitTime
        }));

        const model = await trainModel(processedTrainingData);
        if (!model) { return; }

        console.log('実データ（メッセージのみ）を取得し、予測を実行中...');
        const actualResponse = await fetch(ACTUAL_DATA_URL_MESSAGES_ONLY);
        const actualMessagesOnly = await actualResponse.json();

        // 予測の実行
        const predictedWaitTimes = [];
        const labelsForChart = [];
        actualMessagesOnly.forEach((message, index) => {
            const encodedMessage = encode(message);
            const prediction = model.predict(tf.tensor2d([encodedMessage]));
            predictedWaitTimes.push(Math.round(prediction.dataSync()[0]));
            labelsForChart.push(`Message ${index + 1}`);
        });

        console.log("予測された待ち時間:", predictedWaitTimes);

        // 平均値の計算
        const averagePredictedWaitTime = predictedWaitTimes.reduce((sum, val) => sum + val, 0) / predictedWaitTimes.length;

        // Chart.jsで可視化
        const ctx = document.getElementById('myChart').getContext('2d');
        if (window.myChartInstance) {
            window.myChartInstance.destroy();
        }

        window.myChartInstance = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labelsForChart,
                datasets: [{
                    label: '予測された待ち時間',
                    data: predictedWaitTimes,
                    borderColor: 'rgba(255, 99, 132, 1)',
                    borderWidth: 2,
                    fill: false,
                    tension: 0.1
                }]
            },
            options: {
                // ... (scalesやpluginsの設定は前回のコードと同様)
                plugins: {
                    annotation: {
                        annotations: {
                            line1: {
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
// 実行
run();