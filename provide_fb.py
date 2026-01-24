from app.settings import build_settings

def main():
    print("Hello world!")
    cfg = build_settings()
    # print(cfg.ged.model_name)


if __name__ == "__main__":
    main()