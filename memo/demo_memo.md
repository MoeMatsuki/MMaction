1. demo.pyをコピー
    ```
    cp mmaction2/demo/demo.py demo.py
    ```
2. pathを通すためにsetup.pyを追加
    - [setup.py](setup.py)
    - demo.pyでimportする行を追加
3. なんかエラーが起きたので修正
    ```
    from mmaction.apis import inference_recognizer, init_recognizer
    ↓
    from mmaction.apis.inference import inference_recognizer, init_recognizer
    ```

以上