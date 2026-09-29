# Espelho facial — rosto real

Esta versão substitui o avatar LAM visível por uma composição baseada na imagem real da webcam.

## Destaques desta versão (Milestone)

- **Região dos Olhos:** O espelhamento da região ocular ficou **muito bom nesta versão**, permitindo sincronização perfeita do piscar (blefaroespasmo/lagolftalmo na reabilitação facial), preservando íris, pálpebra e sobrancelha com textura 100% real da webcam.
- **Malha 3D Anatômica (WebGL):** 449 triângulos por hemiface (468 vértices) derivados da topologia canônica MediaPipe/FLAME, mantendo a perspectiva 3D e fidelidade geométrica mesmo quando a pessoa vira a cabeça para os lados (rotação de *yaw* e *pitch*).
- **Modos de Espelhamento:**
  - *Malha 3D Anatômica:* Acompanha a perspectiva 3D volumétrica da cabeça virando de lado.
  - *Espelho Sagital 2D:* Reflexão contínua com suavização de borda (*feathering*).
- **Controle Dinâmico de Hemiface:** Seletor intuitivo permitindo alternar a hemiface saudável (esquerda ou direita) e controle de preservação de olho original vs. sincronização de piscar.

## Executar

```sh
npm ci
npm run dev
```

Abra o endereço `http://localhost:4173` no navegador. A câmera exige `localhost` ou HTTPS; abrir o arquivo diretamente como `file://` pode bloquear a permissão. O Face Landmarker e o modelo são carregados do CDN na primeira ativação da câmera, portanto é necessário acesso à internet nesse momento.

## Protocolo da próxima validação

1. Testar rosto neutro frontal.
2. Testar sorriso, sobrancelha e fechamento de olho separadamente.
3. Ativar “Mostrar malha de suporte” para verificar o alinhamento dos landmarks.
4. Ativar e desativar “Preservar olho direito original” e observar a íris/esclera.
5. Repetir com rotações progressivas até aproximadamente ±30°.
6. Registrar costura, tremor, atraso e FPS em cada condição.
