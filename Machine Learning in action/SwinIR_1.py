model.train()
best_loss = 9999.0
for epoch in range(num_epochs):
    start_time = time.time()
    running_loss = 0.0
    for noisy_images, clean_images in train_loader:
        noisy_images = noisy_images.to(device)
        clean_images = clean_images.to(device)
        optimizer.zero_grad()
        outputs = model(noisy_images)
        resized_image = F.interpolate(outputs, scale_factor=0.5, mode='bilinear', align_corners=False)
        outputs = resized_image
        loss = criterion(outputs, noisy_images-clean_images)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * noisy_images.size(0)
    epoch_loss = running_loss / len(train_dataset)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

# 현재 epoch의 loss가 최소 loss보다 작으면 모델 갱신
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(model.state_dict(), 'best_model2.pth')
        print(f"{epoch+1}epoch 모델 저장 완료")
    # 종료 시간 기록
    end_time = time.time()
    # 소요 시간 계산
    training_time = end_time - start_time
    # 시, 분, 초로 변환
    minutes = int(training_time // 60)
    seconds = int(training_time % 60)
    hours = int(minutes // 60)
    minutes = int(minutes % 60)
    # 결과 출력
    print(f"훈련 소요 시간: {hours}시간 {minutes}분 {seconds}초")