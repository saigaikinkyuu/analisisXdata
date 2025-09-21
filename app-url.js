async function loadJSON(path) {
    const response = await fetch(path);
    return await response.json();
}

// 単純なテキストをベクトル化（Bag of Words風の手法）
function tokenizeMessages(messages, vocab) {
    return messages.map(msg => {
        const tokens = Array(vocab.length).fill(0);
        if (typeof msg === 'string') {
            msg.toLowerCase().split(/[\s!?。、「」]/).forEach(word => {
                const index = vocab.indexOf(word);
                if (index >= 0) tokens[index] = 1;
            });
        }
        return tokens;
    });
}

function buildVocabulary(messages) {
    const vocabSet = new Set();
    messages.forEach(msg => {
        if (typeof msg === 'string') {
            msg.toLowerCase().split(/[\s!?。、「」]/).forEach(word => {
                if (word) vocabSet.add(word);
            });
        }
    });
    return Array.from(vocabSet);
}

async function main() {
  const trainingData = await loadJSON('./json/training_data.json');
  const actualData = await loadJSON('./json/actual_data.json');

  // 999999 は "不明" として扱うが、学習には使用する
  const validTrainingData = trainingData.filter(d =>
    typeof d.message === 'string' &&
    !isNaN(parseFloat(d.waitTime))
  );

  const trainingMessages = validTrainingData.map(d => d.message);
  const trainingLabels = validTrainingData.map(d => parseFloat(d.waitTime));

  const actualMessages = actualData.map(d => d.message);

  const vocab = buildVocabulary(trainingMessages);
  const xs = tf.tensor2d(tokenizeMessages(trainingMessages, vocab));
  const ys = tf.tensor2d(trainingLabels, [trainingLabels.length, 1]);

  // モデル構築と学習
  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 16, activation: 'relu', inputShape: [vocab.length] }));
  model.add(tf.layers.dense({ units: 8, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 1 }));

  model.compile({ loss: 'meanSquaredError', optimizer: 'adam' });
  await model.fit(xs, ys, {
    epochs: 50,
    batchSize: 4,
    shuffle: true
  });

  // 実データで推論
  const actualInputs = tf.tensor2d(tokenizeMessages(actualMessages, vocab));
  const predictions = await model.predict(actualInputs).array();

  actualMessages.forEach((msg, i) => {
    const predicted = predictions[i][0];
    console.log(`メッセージ: "${msg}" → 予測待ち時間: ${predicted === 999999 ? '不明' : predicted.toFixed(2)} 分`);
  });

  // 999999（不明）でないものだけチャートに表示
  const filteredResults = actualMessages.map((msg, i) => {
    return { message: msg, waitTime: predictions[i][0] };
  }).filter(result => result.waitTime !== 999999);

  // Chart.js 描画
  const ctx = document.getElementById('waitChart').getContext('2d');
  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: filteredResults.map(r => r.message),
      datasets: [{
        label: '予測待ち時間（分）',
        data: filteredResults.map(r => r.waitTime),
        backgroundColor: 'rgba(75, 192, 192, 0.6)'
      }]
    },
    options: {
      responsive: true,
      scales: {
        y: {
          beginAtZero: true,
          title: { display: true, text: '待ち時間（分）' }
        },
        x: {
          title: { display: true, text: 'メッセージ' }
        }
      }
    }
  });
}

main();
