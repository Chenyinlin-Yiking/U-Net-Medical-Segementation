from utils.run import just_do_it
from utils.decorator_time import display_time


@display_time(text="total consume")
def main():

    """
    主函数
    :return: 0
    """
    just_do_it()


if __name__ == '__main__':
    main()
